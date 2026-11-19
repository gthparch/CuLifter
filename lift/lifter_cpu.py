import os
import re
from functools import partial

from llvmlite import binding, ir

from transform.predicated_to_sel import PredicatedToSel
from transform.mem_index_fold import MemIndexFold
from transform.desc_const_map import RecordDescriptorConsts
from transform.wgmma_swizzle import RecordWgmmaSwizzle
from transform.build_cfg import BuildCFG
from transform.resolve_indirect_call import ResolveIndirectCall
from transform.handle_call import HandleCall
from transform.cg_patterns import CGPatterns
from transform.warp_patterns import WarpPatterns
from transform.sr_substitute import SR_TO_OFFSET, SRSubstitute
from transform.opmod_transform import OpModTransform
from transform.defuse_analysis import DefUseAnalysis
from transform.ssa import SSA
from transform.reg_remap import RegRemap
from transform.xmad_to_imad import XmadToImad
from transform.fp_hack import FPHack
from transform.set_zero import SetZero
from transform.dce import DCE
from transform.mov_eliminate import MovEliminate
from transform.typeanalysis import TypeAnalysis, FLOAT32, FLOAT64
from transform.resolve_brx import ResolveBRX
from transform.cfg_simplify import CFGSimplify
from transform.intdiv_recognize import IntDivRecognize
from transform.tma_coord_recover import TmaCoordRecover
from lift.lifter import Lifter, lift_for, UnsupportedInstructionException

# Opt-in lockstep mode: bracket shared-memory accesses with warp_turn_begin/end
# runtime calls so pre-Volta warp-synchronous code (volatile shared, no
# __syncwarp) executes in SIMT round order under the pthread wrapper. Default
# off => emitted IR is byte-identical to before.
_LOCKSTEP = os.environ.get("CULIFTER_LOCKSTEP") == "1"


class UnsupportedOperatorException(Exception):
    pass


class InvalidTypeException(Exception):
    pass


class CpuLifter(Lifter):
    # Sentinel for the high 32 bits of SM90 generic shared-memory addresses
    # (SR_SWINHI). Real x86 heap allocations never sit at this high half, so
    # generic LD/ST can recognize lifted shared-window pointers and redirect
    # them into the shared_mem array (see _redirect_shared_window).
    SHARED_WINDOW_TAG = 0x42424242
    # Warp collectives carry an explicit site id (the SASS instruction
    # address) so the runtime's history matching survives clang call-site
    # duplication (see Lifter.collective_site_arg).
    collective_site_arg = True

    # The x86 backend runs a single CTA (cluster-of-1), so Hopper's
    # cluster-shared (DSMEM) `mapa` lowering must be IDENTITY: a DSMEM self-address
    # aliases the local shared window. lift_prmt uses this to collapse the mapa
    # base (the `PRMT rank,0x654,window_base` -> (rank<<24)|window_base pattern) to
    # 0 in cluster-bulk-copy kernels, so UBLKCP/UBLKRED/mbarrier addresses land at
    # the SAME @shared_mem offset that local LDS/STS read. (nvvm targets real GPU
    # clusters and leaves this False.)
    _cluster_is_local = True

    # Serialized-warp backend (--lifter x86warp): when True, the lift targets the
    # one-OS-thread-per-warp fiber runtime — R2UR (and other reg->uniform/elected
    # publishes) write a per-warp shared uniform slot, const_mem/local_mem are
    # per-lane, and collectives yield. Off (default) = the standard per-CUDA-thread
    # x86/arm lift, byte-identical to before. See
    # Build is incremental: this flag
    # is the entry point; the warp-mode emission lands milestone-by-milestone.
    # Set by main.py for `--lifter x86warp`, but currently READ NOWHERE: the
    # serialized-warp model became unconditional, so x86 and x86warp emit
    # byte-identical IR. The CLI choice is retained for harness compatibility.
    warp_mode = False

    def get_transform_passes(self):
        # Ablation support: SLIFTER_SKIP_PASSES=Foo,Bar skips named passes.
        # SLIFTER_TYPE_SEED_ONLY=1 runs TypeAnalysis with max_iterations=0.
        skip_passes = set(
            os.environ.get("SLIFTER_SKIP_PASSES", "").split(",")
        ) - {""}
        seed_only = os.environ.get("SLIFTER_TYPE_SEED_ONLY", "") == "1"

        passes = [
            BuildCFG(),
            ResolveIndirectCall(),
            ResolveBRX(),
            HandleCall(),
            CFGSimplify(),
            CGPatterns(),
            WarpPatterns(),
            SRSubstitute(),
            OpModTransform(),
            PredicatedToSel(),
            # Attach the 4D TMA tensor coordinates (implicit UR window) as
            # explicit use operands of each UTMALDG.4D, recovered by reaching-def
            # over the CFG, BEFORE SSA so they are versioned to the correct
            # per-instruction reaching definition and survive DCE/RegRemap.
            # This is THE FlashAttention 4D-coord blocker fix.
            TmaCoordRecover(),
            # Fold two-register memory addresses [Rbase+Rindex] into a tracked
            # IADD3 so the index register (e.g. the SM90 shared-window mbarrier
            # offset) survives SSA/DCE and the lifter resolves the full address.
            MemIndexFold(),
            # Record descriptor-register const loads (ULDC.64 URx, c[..]) under
            # their pre-rename names so _resolve_descriptor_addr can recover a
            # desc[URx] base after SSA/RegRemap rename + DCE the defining load.
            RecordDescriptorConsts(),
            # Record each descriptor-form HGMMA/WGMMA's per-operand GMMA swizzle
            # mode (SW128/SW64/SW32) from the descriptor HIGH-word layout immediate
            # under pre-rename UR names, so the lifter reads the right atom width
            # per operand (FA3 hd96 mixes Q/dO SW64 + K SW32). Emits no IR; SW128
            # kernels are byte-identical.
            RecordWgmmaSwizzle(),
            SSA(),
            RegRemap(),
            DefUseAnalysis(),       # #1: needed by FPHack (reads users)
            # Collapse the SASS Granlund-Möller magic-divide sequence (anchored
            # by IADD3 +0x0FFFFFFE off MUFU.RCP) into a single SDIV pseudo-op
            # (lifted as udiv). On the CPU backend the raw chain runs a software
            # rcp_f32 whose bits don't match the hardware MUFU.RCP the magic
            # number is tuned for, yielding wholly wrong quotients (e.g. nvjpeg's
            # per-image index). Needs SSA + def-use for the reaching-defs walk;
            # splices instructions, so refresh def-use immediately after.
            IntDivRecognize(),
            DefUseAnalysis(),       # #1b: refresh after IntDivRecognize splice
            XmadToImad(),
            FPHack(),               # reads users from DefUse#1
            RegRemap(),
            SetZero(),
            DefUseAnalysis(),       # #2: needed by MovEliminate (reads users)
            MovEliminate(),         # reads users from DefUse#2
            DefUseAnalysis(),       # #3: needed by DCE (reads reaching_defs)
            DCE(),                  # reads reaching_defs from DefUse#3
            CFGSimplify(),
            # DefUse#4 REMOVED — only RegRemap follows, which doesn't read def-use
            RegRemap(),
            DefUseAnalysis(),       # #4: needed by DCE#2 (reads reaching_defs)
            DCE(),                  # reads reaching_defs from DefUse#4
            CFGSimplify(),
            # DefUse#6 REMOVED — only TypeAnalysis follows, which doesn't read def-use
            TypeAnalysis(max_iterations=0) if seed_only else TypeAnalysis(),
            # DefUse#7 REMOVED — only RegRemap follows, which doesn't read def-use
            RegRemap(),
        ]

        if skip_passes:
            passes = [p for p in passes
                      if type(p).__name__ not in skip_passes]

        return passes

    def setup_module(self, llvm_module):
        shared_array_ty = self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.SharedMem = self.ir.GlobalVariable(
            llvm_module, shared_array_ty, "shared_mem"
        )
        local_array_ty = self.ir.ArrayType(self.ir.IntType(8), 32768)
        self.LocalMem = self.ir.GlobalVariable(llvm_module, local_array_ty, "local_mem")
        self._setup_nv_globals(llvm_module)

    def _setup_nv_globals(self, llvm_module):
        """Materialize the cubin's device-global sections.

        `.nv.global.init` (PROGBITS) and `.nv.global` (NOBITS BSS) hold
        __device__ globals — e.g. curand's precomputed XORWOW skip-ahead
        matrices. SASS reaches them through loader relocations: 64-bit
        addresses in const bank 4 (R_CUDA_64 — see
        _maybe_emit_nvglobal_const4_ctor) or 32-bit halves patched into
        `UMOV URx, 0x0` immediates (R_CUDA_ABS32_LO/HI_32 — see lift_mov).
        Without them every table base reads 0 and the kernel faults.
        Keyed by in_init_section (True = .nv.global.init).
        """
        self._nv_globals = {}
        self._nv_global_ctor_done = False
        elf = getattr(self, "elf", None)
        if elf is None:
            return
        i8 = self.ir.IntType(8)
        init_bytes = getattr(elf, "nv_global_init", b"") or b""
        if init_bytes:
            arr_ty = self.ir.ArrayType(i8, len(init_bytes))
            g = self.ir.GlobalVariable(
                llvm_module, arr_ty, "__slifter_nv_global_init")
            g.initializer = self.ir.Constant(arr_ty, bytearray(init_bytes))
            g.linkage = "internal"
            g.align = 16
            self._nv_globals[True] = g
        bss_size = getattr(elf, "nv_global_size", 0) or 0
        if bss_size:
            arr_ty = self.ir.ArrayType(i8, bss_size)
            g = self.ir.GlobalVariable(
                llvm_module, arr_ty, "__slifter_nv_global")
            g.initializer = self.ir.Constant(arr_ty, None)
            g.linkage = "internal"
            g.align = 16
            self._nv_globals[False] = g

    def _nv_global_addr(self, builder, symname, name="nvg"):
        """i64 host address of a device-global symbol, or None."""
        elf = getattr(self, "elf", None)
        rec = (getattr(elf, "nv_global_syms", None) or {}).get(symname)
        if rec is None:
            return None
        in_init, off, _size = rec
        g = self._nv_globals.get(in_init)
        if g is None:
            return None
        i64 = self.ir.IntType(64)
        addr = builder.ptrtoint(g, i64, f"{name}_addr")
        if off:
            addr = builder.add(
                addr, self.ir.Constant(i64, off), f"{name}_addr_off")
        return addr

    def _maybe_emit_nvglobal_const4_ctor(self, llvm_module):
        """Fill const bank 4's relocation slots with live global addresses.

        The CUDA loader writes each `.rela.nv.constant4` R_CUDA_64 symbol
        address into its c[0x4] slot before launch. Reproduce that with a
        global ctor storing into runtime const_mem[4] — the ctor runs on the
        main thread, whose const_mem launchKernel snapshots into
        g_constMemTemplate for the worker threads. Host drivers that fill
        c[0x4] slots do so in main(), after ctors, so they still win.
        """
        if self._nv_global_ctor_done or self.ConstMem is None:
            return
        elf = getattr(self, "elf", None)
        relocs = getattr(elf, "const4_relocs", None) or {}
        if not relocs or not self._nv_globals:
            return
        syms = getattr(elf, "nv_global_syms", None) or {}
        slots = [(slot, name)
                 for slot, (name, _v, is_func) in sorted(relocs.items())
                 if not is_func and name in syms]
        if not slots:
            return
        self._nv_global_ctor_done = True

        i8p = self.ir.IntType(8).as_pointer()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        fnty = self.ir.FunctionType(self.ir.VoidType(), [])
        ctor = self.ir.Function(llvm_module, fnty, "__slifter_nv_const4_init")
        ctor.linkage = "internal"
        builder = self.ir.IRBuilder(ctor.append_basic_block("entry"))
        for slot, name in slots:
            addr = self._nv_global_addr(builder, name, f"c4_{slot:x}")
            ptr = builder.gep(self.ConstMem, [
                self.ir.Constant(i64, 0),
                self.ir.Constant(i64, 4),
                self.ir.Constant(i64, slot),
            ], f"c4_slot_{slot:x}")
            ptr = builder.bitcast(
                ptr, self.ir.PointerType(i64, self.CONST_MEM_ADDRSPACE))
            builder.store(addr, ptr)
        builder.ret_void()

        elem_ty = self.ir.LiteralStructType([i32, fnty.as_pointer(), i8p])
        entry = self.ir.Constant.literal_struct([
            self.ir.Constant(i32, 65535), ctor, self.ir.Constant(i8p, None)])
        arr_ty = self.ir.ArrayType(elem_ty, 1)
        gctors = self.ir.GlobalVariable(
            llvm_module, arr_ty, "llvm.global_ctors")
        gctors.linkage = "appending"
        gctors.initializer = self.ir.Constant(arr_ty, [entry])

    def _abs32_addr_half(self, Inst, IRBuilder):
        """R_CUDA_ABS32_LO/HI_32: this MOV/UMOV's zero immediate is patched
        by the loader with one 32-bit half of a device-global's address.
        inst.id is the SASS text offset — the relocation key."""
        elf = getattr(self, "elf", None)
        if elf is None or not getattr(self, "_nv_globals", None):
            return None
        relmap = (getattr(elf, "rel_text_abs32", None) or {}).get(
            self.curr_function.name)
        if not relmap:
            return None
        try:
            off = int(str(Inst.id).split("_")[0].split("@")[0], 16) & ~0xF
        except ValueError:
            return None
        ent = relmap.get(off)
        if ent is None:
            return None
        symname, is_hi = ent
        addr = self._nv_global_addr(IRBuilder, symname, "nvg")
        if addr is None:
            return None
        if is_hi:
            addr = IRBuilder.lshr(
                addr, self.ir.Constant(self.ir.IntType(64), 32), "nvg_hi")
        return IRBuilder.trunc(addr, self.ir.IntType(32), "nvg_half")

    @lift_for("MOV", "MOV64", "UMOV", "R2UR")
    def lift_mov(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None,
                 **kwargs):
        half = self._abs32_addr_half(Inst, IRBuilder)
        if half is not None:
            self._set_val(IRBuilder, IRRegs, Inst.get_defs()[0], half)
            return
        return super().lift_mov(Inst, IRBuilder, IRRegs, block_map=block_map,
                                _get_val=_get_val, **kwargs)

    def _setup_const_mem(self, llvm_module, func_name):
        if self.ConstMem is not None:
            return
        bank_size = 4096
        bank_ty = self.ir.ArrayType(self.ir.IntType(8), bank_size)
        const_array_ty = self.ir.ArrayType(bank_ty, self.NUM_CONST_MEM_BANKS)
        self.ConstMem = self.ir.GlobalVariable(
            llvm_module, const_array_ty, "const_mem", self.CONST_MEM_ADDRSPACE
        )

        self._elf_const_banks = {}
        if getattr(self, "elf", None) is not None:
            # const_words falls back to module-shared banks (bare
            # `.nv.constant<N>`, no kernel suffix) — e.g. lci-cuda's factorial
            # tables in bank 3. Reading elf_const_data[func_name] directly
            # missed those, leaving the bank zeroed (NaN cluster).
            #
            # Bank 4 is the device-function/global pointer table: its ELF
            # bytes are pre-relocation zeros, and the host driver populates
            # runtime const_mem[4] with live pointers (host_gen's c[0x4]
            # detection). Materializing it statically made kernels read NULL
            # instead — 8 HeCBench kernels (collision/fft/gabor/zmddft/…)
            # regressed pass -> SIGSEGV. It must stay runtime, like bank 0.
            for bank in range(1, min(4, self.NUM_CONST_MEM_BANKS)):
                words = self.elf.const_words(func_name, bank)
                if words:
                    elf_bank_size = len(words) * 4
                    if elf_bank_size > 0:
                        i8 = self.ir.IntType(8)
                        bank_bytes = []
                        for word in words:
                            for b in range(4):
                                bank_bytes.append(self.ir.Constant(i8, (word >> (b * 8)) & 0xFF))
                        # Pad to the hardware const-bank window (64 KB): SASS
                        # may legally index past the section's data and the GPU
                        # returns 0 there — lci-cuda's alpha() reads
                        # double_fact_table past .nv.constant3's 3752 bytes and
                        # depends on the inf-producing zeros. An unpadded
                        # global makes those reads UB (heap garbage -> NaN).
                        zero = self.ir.Constant(i8, 0)
                        bank_bytes.extend(
                            [zero] * (self.CONST_BANK_WINDOW - len(bank_bytes)))
                        bank_arr_ty = self.ir.ArrayType(i8, len(bank_bytes))
                        elf_bank = self.ir.GlobalVariable(
                            llvm_module, bank_arr_ty, f"const_mem_bank{bank}"
                        )
                        elf_bank.initializer = self.ir.Constant.literal_array(bank_bytes)
                        elf_bank.global_constant = True
                        self._elf_const_banks[bank] = (elf_bank, len(bank_bytes))

        self._maybe_emit_nvglobal_const4_ctor(llvm_module)

    def _should_load_const_from_memory(self, bank, byte_offset):
        # Bank 0 holds kernel args + SRs that the host populates at launch
        # time — those must come from runtime const_mem, not the static
        # initializer. The one exception is c[0x0][0x28], which carries the
        # per-kernel EIATTR_FRAME_SIZE (from the cubin's .nv.info). The
        # SASS prologue computes its local-memory base as
        # `LDC R1, c[0x0][0x28]; VIADD R1, R1, -frame_size`; if the load
        # comes back as 0 (the default for a host-defined @const_mem) the
        # base underflows in u32 and every LDL/STL goes out of bounds.
        # Let the parent's _load_const_from_elf_const_data substitute the
        # known constant for any load whose 4-byte range starts at 0x28.
        if bank == 0 and byte_offset != 0x28:
            return True
        # Bank 4 is the loader-relocated pointer table. When the nv-global
        # ctor fills its slots at runtime, static folds of the section's
        # pre-relocation zeros would bypass those stores.
        if bank == 4 and getattr(self, "_nv_global_ctor_done", False):
            return True
        return False

    def _load_from_const_mem(self, builder, bank, byte_offset, target_ty, name=""):
        if bank > 0 and bank in self._elf_const_banks:
            elf_bank, size = self._elf_const_banks[bank]
            if byte_offset < size:
                ptr = builder.gep(
                    elf_bank,
                    [
                        self.ir.Constant(self.ir.IntType(64), 0),
                        self.ir.Constant(self.ir.IntType(64), byte_offset),
                    ],
                    f"{name}_ptr" if name else "elf_cm_ptr",
                )
                ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty))
                return builder.load(ptr, name if name else "elf_cm_load")
        
        const_val = self._load_const_from_elf_const_data(bank, byte_offset, target_ty)
        if const_val is not None:
            return const_val

        ptr = builder.gep(
            self.ConstMem,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), bank),
                self.ir.Constant(self.ir.IntType(64), byte_offset),
            ],
            f"{name}_ptr" if name else "cm_ptr",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty, self.CONST_MEM_ADDRSPACE))
        return builder.load(ptr, name if name else "cm_load")

    def _load_from_const_mem_dynamic(self, builder, bank, byte_offset_val, target_ty, name=""):
        # Register-indexed reads of module const banks (`LDC.64 Rd, c[0x3][Rx]`,
        # e.g. lci-cuda's factorial tables) must hit the materialized
        # const_mem_bank<N> global, same as the static path above — the runtime
        # @const_mem banks >0 are zero-initialized and never populated by the
        # host, so falling through to them silently reads 0 (NaN cluster).
        if bank > 0 and bank in self._elf_const_banks:
            elf_bank, _size = self._elf_const_banks[bank]
            if byte_offset_val.type != self.ir.IntType(64):
                if byte_offset_val.type.width < 64:
                    byte_offset_val = builder.zext(
                        byte_offset_val, self.ir.IntType(64), f"{name}_off_zext")
                else:
                    byte_offset_val = builder.trunc(
                        byte_offset_val, self.ir.IntType(64), f"{name}_off_trunc")
            ptr = builder.gep(
                elf_bank,
                [self.ir.Constant(self.ir.IntType(64), 0), byte_offset_val],
                f"{name}_ptr" if name else "elf_cm_dyn_ptr",
            )
            ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty))
            return builder.load(ptr, name if name else "elf_cm_dyn_load")
        return super()._load_from_const_mem_dynamic(
            builder, bank, byte_offset_val, target_ty, name)

    def _get_intrinsic(self, name):
        if name in self.intrinsics:
            return self.intrinsics[name]

        i1 = self.ir.IntType(1)
        i8 = self.ir.IntType(8)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        f32 = self.ir.FloatType()
        f64 = self.ir.DoubleType()

        func = None
        if name == "abs":
            func_ty = self.ir.FunctionType(i32, [i32, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.abs.i32")
        elif name == "memcpy_i64":
            # llvm.memcpy(dst i8*, src i8*, len i64, isvolatile i1)
            i8p = self.ir.PointerType(i8)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i8p, i8p, i64, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.memcpy.p0i8.p0i8.i64")
        elif name == "fabs":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.fabs.f32")
        elif name == "fabs_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.fabs.f64")
        elif name == "dmma884f64":
            # DMMA.8x8x4 (m8n8k4 F64) lane-0 scalar FMA.
            # Args: (A, B, C0, C1) — all f64.
            # Per PTX mma.sync.m8n8k4.f64: thread i holds A[i/4, i%4],
            # B[i%4, i/4], C[2*(i/4)..2*(i/4)+1, i%4].
            # Lane 0: A[0,0], B[0,0], C[0..1, 0] → two D slots:
            #   d0 = c0 + A*B   (row 0 col 0, lane-0 k=0 contribution)
            #   d1 = c1          (row 1 col 0, lane-0 has no A[1,0])
            # Other lanes (1..31) contribute their own k and row/col slice;
            # on single-thread x86 we emit only lane 0's share.
            output_ty = ir.LiteralStructType([f64] * 2)
            func_ty = ir.FunctionType(output_ty, [f64] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name="dmma884f64")
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            A, B, C0, C1 = func.args[0], func.args[1], func.args[2], func.args[3]
            prod = b.fmul(A, B, "dmma_p0")
            d0 = b.fadd(C0, prod, "dmma_d0_fma")
            d1 = C1
            out = ir.Constant(output_ty, None)
            out = b.insert_value(out, d0, 0, "dmma_d0")
            out = b.insert_value(out, d1, 1, "dmma_d1")
            b.ret(out)
        elif name.startswith("imma_"):
            # x86 single-thread pass-through for integer tensor-core MMA.
            # Supports dense (imma_m*n*k*_{...}_sN_sN) and sparse 2:4
            # (imma_m16n8k64_..._sparse). Body is C → D identity so the
            # surrounding IR remains type-correct; actual per-lane reduction
            # is not modeled on x86.
            # Shape is embedded in the name (first token after "imma_"):
            #   m8n8k16  → d=2, a=1, b=1, c=2 → 2 out, 4 in
            #   m8n8k32  → d=2, a=1, b=1, c=2 → 2 out, 4 in
            #   m16n8k16 → d=4, a=2, b=1, c=4 → 4 out, 7 in
            #   m16n8k32 → d=4, a=4, b=2, c=4 → 4 out, 10 in
            #   m16n8k64 → d=4, a=4, b=4, c=4 → 4 out, 12 in (sparse;
            #              metadata+selector dropped before callsite)
            shape_tok = name[len("imma_"):].split("_", 1)[0]
            imma_shapes = {
                "m8n8k16":  (2, 4),
                "m8n8k32":  (2, 4),
                "m16n8k16": (4, 7),
                "m16n8k32": (4, 10),
                "m16n8k64": (4, 12),
            }
            n_out, n_in = imma_shapes.get(shape_tok, (4, 12))
            output_ty = ir.LiteralStructType([i32] * n_out)
            func_ty = ir.FunctionType(output_ty, [i32] * n_in, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            out = ir.Constant(output_ty, None)
            # Trailing operands are always the C accumulators.
            for j in range(n_out):
                out = b.insert_value(out, func.args[-n_out + j],
                                     j, f"imma_c{j}")
            b.ret(out)
        elif name == "hmma1688f32_tf32":
            # HMMA.1688.F32.TF32: warp-true via the wrapper rendezvous
            # (hmma1688_tf32_warp) — the standard Ampere f32 tensor path
            # (cutlass ex14/27/28/29/33/59, cuBLAS tf32 sgemm). m16n8k8 with
            # ONE tf32 per full 32-bit reg: A=4 regs, B=2 regs, C/D=4 f32
            # (cute MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>). The runtime
            # masks each A/B mantissa to tf32 (low 13 bits zeroed — SASS
            # reads raw regs, the tensor core ignores them) before the f32
            # FMA. Args: f32 a0..a3, b0, b1, c0..c3 -> {f32 x4}.
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(output_ty, [f32] * 10, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            f32ptr = ir.PointerType(f32)
            ext_ty = ir.FunctionType(
                ir.VoidType(), [f32ptr] + [f32] * 10, False)
            ext = self.ir_functions.get("hmma1688_tf32_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma1688_tf32_warp")
                self.ir_functions["hmma1688_tf32_warp"] = ext
            dvec = b.alloca(ir.ArrayType(f32, 4), name="hmma_d")
            dptr = b.bitcast(dvec, f32ptr, "hmma_dptr")
            b.call(ext, [dptr] + list(func.args))
            out = ir.Constant(output_ty, None)
            for k in range(4):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                out = b.insert_value(out, b.load(p, f"d{k}"), k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma1688f32":
            # HMMA.1688.F32: warp-true via the wrapper rendezvous
            # (hmma1688_f32_warp) — m16n8k8 f16-input/f32-accum (cutlass
            # ex12 gemm_bias_relu, ex41 attention). Fragment layout is the
            # k<8 half of 16816 (A: 2 regs, B: 1 reg, same f32 C/D ABI).
            # The old lane-0 stub left each lane its own 2-of-8 k-product
            # partial with d1/d3 accumulator pass-throughs (hmma1688f32
            # fixture: 128/128 mismatch).
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(output_ty, [v2f16] * 3 + [f32] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            f32ptr = ir.PointerType(f32)
            ext_ty = ir.FunctionType(
                ir.VoidType(), [f32ptr] + [i32] * 3 + [f32] * 4, False)
            ext = self.ir_functions.get("hmma1688_f32_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma1688_f32_warp")
                self.ir_functions["hmma1688_f32_warp"] = ext
            dvec = b.alloca(ir.ArrayType(f32, 4), name="hmma_d")
            dptr = b.bitcast(dvec, f32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(3):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            for k in range(3, 7):
                call_args.append(func.args[k])
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(4):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                out = b.insert_value(out, b.load(p, f"d{k}"), k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma1688f32_bf16":
            # HMMA.1688.F32.BF16: warp-true via the wrapper rendezvous
            # (hmma1688_bf16_warp) — m16n8k8 bf16-input/f32-accum (cutlass
            # ex41 backward, 160x). Identical staging/finalize ABI to
            # hmma1688_f32_warp; only the A/B fragment decode differs
            # (packed bf16 pairs instead of f16 pairs). The old lane-0 stub
            # gave 128/128 mismatch on the hmma1688bf16 fixture.
            from sir.bf16_type import BFloat16Type
            v2bf16 = ir.VectorType(BFloat16Type(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(output_ty, [v2bf16] * 3 + [f32] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            f32ptr = ir.PointerType(f32)
            ext_ty = ir.FunctionType(
                ir.VoidType(), [f32ptr] + [i32] * 3 + [f32] * 4, False)
            ext = self.ir_functions.get("hmma1688_bf16_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma1688_bf16_warp")
                self.ir_functions["hmma1688_bf16_warp"] = ext
            dvec = b.alloca(ir.ArrayType(f32, 4), name="hmma_d")
            dptr = b.bitcast(dvec, f32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(3):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            for k in range(3, 7):
                call_args.append(func.args[k])
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(4):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                out = b.insert_value(out, b.load(p, f"d{k}"), k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma1688f16":
            # HMMA.1688.F16: warp-true via the wrapper rendezvous
            # (hmma1688_f16_warp) — m16n8k8 F16-accum (cutlass ex13 fused
            # gemm/conv chains, the most numerous 1688 flavor). Same A/B
            # layout as hmma1688_f32_warp; C arrives as 2 packed v2f16 regs
            # (4 halves) and D returns the same way. The old lane-0 stub
            # left each lane its own 2-of-8 k-product partial (hmma1688f16
            # fixture: 128/128 mismatch).
            # Args: v2f16 a0, a1, b0, c0, c1 -> {v2f16, v2f16}
            half_ty = ir.HalfType()
            v2f16 = ir.VectorType(half_ty, 2)
            output_ty = ir.LiteralStructType([v2f16] * 2)
            func_ty = ir.FunctionType(output_ty, [v2f16] * 5, False)
            func = ir.Function(self.llvm_module, func_ty, name="hmma1688f16")
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            i32ptr = ir.PointerType(i32)
            ext_ty = ir.FunctionType(ir.VoidType(), [i32ptr] + [i32] * 5, False)
            ext = self.ir_functions.get("hmma1688_f16_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma1688_f16_warp")
                self.ir_functions["hmma1688_f16_warp"] = ext
            dvec = b.alloca(ir.ArrayType(i32, 2), name="hmma_d")
            dptr = b.bitcast(dvec, i32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(5):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(2):
                pp = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                           f"d{k}_ptr")
                out = b.insert_value(
                    out, b.bitcast(b.load(pp, f"d{k}_i32"), v2f16, f"d{k}"),
                    k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma16816f32":
            # HMMA.16816.F32: warp-exchange emulation. The m16n8k16 fragment
            # layout spreads K across each quad and N across groups, so a
            # per-lane decomposition sees only 4 of 16 K products
            # (tensor_simple: 0.667 vs 3.457). Build a thin wrapper with the
            # original signature that routes to the runtime collective
            # hmma16816_f32_warp (kernel_wrapper_75.h), which gathers all 32
            # lanes' fragments and computes the full tile.
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(output_ty, [v2f16] * 6 + [f32] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            f32ptr = ir.PointerType(f32)
            ext_ty = ir.FunctionType(
                ir.VoidType(), [f32ptr] + [i32] * 6 + [f32] * 4, False)
            ext = self.ir_functions.get("hmma16816_f32_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma16816_f32_warp")
                self.ir_functions["hmma16816_f32_warp"] = ext
            dvec = b.alloca(ir.ArrayType(f32, 4), name="hmma_d")
            dptr = b.bitcast(dvec, f32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(6):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            for k in range(6, 10):
                call_args.append(func.args[k])
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(4):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                out = b.insert_value(out, b.load(p, f"d{k}"), k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma16816f32_bf16":
            # HMMA.16816.F32.BF16: warp-true via the wrapper rendezvous
            # (hmma16816_bf16_warp), mirroring the F16-input routing above —
            # same m16n8k16 fragment layout and f32 C/D ABI, only the A/B
            # decode is bf16. The old lane-0 stub here left each lane its
            # own 4-of-16 k-product partial with d1/d3 accumulator
            # pass-throughs (hmma16816bf16 fixture: 126/128 mismatch).
            from sir.bf16_type import BFloat16Type
            v2bf16 = ir.VectorType(BFloat16Type(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(output_ty, [v2bf16] * 6 + [f32] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            f32ptr = ir.PointerType(f32)
            ext_ty = ir.FunctionType(
                ir.VoidType(), [f32ptr] + [i32] * 6 + [f32] * 4, False)
            ext = self.ir_functions.get("hmma16816_bf16_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma16816_bf16_warp")
                self.ir_functions["hmma16816_bf16_warp"] = ext
            dvec = b.alloca(ir.ArrayType(f32, 4), name="hmma_d")
            dptr = b.bitcast(dvec, f32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(6):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            for k in range(6, 10):
                call_args.append(func.args[k])
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(4):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                out = b.insert_value(out, b.load(p, f"d{k}"), k, f"hmma_o{k}")
            b.ret(out)
        elif name == "hmma16816f16":
            # HMMA.16816.F16: warp-true via the wrapper rendezvous
            # (hmma16816_f16_warp), mirroring the F32-accum routing above.
            # The old lane-0 stub left each lane its own k-slice partial —
            # the hgemm tensorop kernel produced coherent ~1/10-scale
            # outputs (every element wrong by the missing cross-lane sum).
            # Args: v2f16 a0..a3, b0, b1, c0, c1 -> {v2f16, v2f16}
            half_ty = ir.HalfType()
            v2f16 = ir.VectorType(half_ty, 2)
            output_ty = ir.LiteralStructType([v2f16] * 2)
            func_ty = ir.FunctionType(output_ty, [v2f16] * 8, False)
            func = ir.Function(self.llvm_module, func_ty, name="hmma16816f16")
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            i32ptr = ir.PointerType(i32)
            ext_ty = ir.FunctionType(ir.VoidType(), [i32ptr] + [i32] * 8, False)
            ext = self.ir_functions.get("hmma16816_f16_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "hmma16816_f16_warp")
                self.ir_functions["hmma16816_f16_warp"] = ext
            dvec = b.alloca(ir.ArrayType(i32, 2), name="hmma_d")
            dptr = b.bitcast(dvec, i32ptr, "hmma_dptr")
            call_args = [dptr]
            for k in range(8):
                call_args.append(b.bitcast(func.args[k], i32, f"frag_{k}_i32"))
            b.call(ext, call_args)
            out = ir.Constant(output_ty, None)
            for k in range(2):
                pp = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                           f"d{k}_ptr")
                out = b.insert_value(
                    out, b.bitcast(b.load(pp, f"d{k}_i32"), v2f16, f"d{k}"),
                    k, f"hmma_o{k}")
            b.ret(out)
        elif name.startswith("ldsm_m88_x") or name.startswith("ldsm_mt88_x"):
            # LDSM.16.[M|MT]88.N (PTX ldmatrix.m8n8[.trans].xN): thin wrapper
            # that parks the lane's 16-byte row in the warp-collective
            # ldsm_m88_warp (kernel_wrapper_75.h) and returns the lane's N
            # redistributed fragment regs packed into one i(N*32). ldmatrix
            # fragments are gathered ACROSS lanes (reg m of lane L = slice
            # (L&3) of the row addressed by lane 8m+(L>>2); MT88 transposed),
            # so a per-lane decomposition cannot be correct.
            trans = 1 if name.startswith("ldsm_mt88") else 0
            count = int(name.rsplit("x", 1)[1])
            ret_ty = self.ir.IntType(count * 32)
            func_ty = ir.FunctionType(ret_ty, [i32] * 4, False)
            func = ir.Function(self.llvm_module, func_ty, name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)
            i32ptr = ir.PointerType(i32)
            ext_ty = ir.FunctionType(ir.VoidType(), [i32ptr] + [i32] * 6, False)
            ext = self.ir_functions.get("ldsm_m88_warp")
            if ext is None:
                ext = ir.Function(self.llvm_module, ext_ty, "ldsm_m88_warp")
                self.ir_functions["ldsm_m88_warp"] = ext
            dvec = b.alloca(ir.ArrayType(i32, 4), name="ldsm_d")
            dptr = b.bitcast(dvec, i32ptr, "ldsm_dptr")
            b.call(ext, [dptr] + list(func.args)
                   + [ir.Constant(i32, count), ir.Constant(i32, trans)])
            out = None
            for k in range(count):
                p = b.gep(dvec, [ir.Constant(i32, 0), ir.Constant(i32, k)],
                          f"d{k}_ptr")
                w = b.load(p, f"d{k}")
                if count > 1:
                    w = b.zext(w, ret_ty, f"d{k}_z")
                    if k:
                        w = b.shl(w, ir.Constant(ret_ty, 32 * k), f"d{k}_s")
                out = w if out is None else b.or_(out, w, f"ldsm_acc{k}")
            b.ret(out)
        elif name.startswith("wgmma64") and name.endswith("f32"):
            # Generalized WGMMA/HGMMA.64xNx16.F32: scalar FMA decomposition
            # for x86 (lane 0, warp 0).  Supports all N in {8,16,32,64,128,256}.
            #
            # Pattern per column-pair cp (0 .. N/8-1):
            #   D[4*cp+0] = C[4*cp+0] + a0_lo*b[2cp]_lo + a0_hi*b[2cp]_hi
            #                          + a2_lo*b[2cp+1]_lo + a2_hi*b[2cp+1]_hi
            #   D[4*cp+1] = C[4*cp+1] + a0_lo*b[2cp]_lo ... (even col of next pair)
            #               — for N=8: only 1 column pair, D1=C1 pass-through
            #               — for N>=16: computed from the next B pair
            #   D[4*cp+2] = C[4*cp+2] + a1/a3 FMA (row 8..15)
            #   D[4*cp+3] = C[4*cp+3] pass-through (odd column)
            #
            # Revised lane-0 mapping: D regs are grouped by column pairs.
            # Each column pair cp uses B regs b[2*cp] and b[2*cp+1].
            # Within each group of 4 D regs:
            #   D[4cp+0]: even col, rows 0-7   -> FMA with a0/a2 and b pair
            #   D[4cp+1]: odd col, rows 0-7    -> pass-through C (lane 0 has no odd col B data)
            #   D[4cp+2]: even col, rows 8-15  -> FMA with a1/a3 and b pair
            #   D[4cp+3]: odd col, rows 8-15   -> pass-through C
            #
            # Extract N from the name: "wgmma64<N>16f32"
            n_str = name[len("wgmma64"):-len("16f32")]
            n_dim = int(n_str)
            a_count = 4
            b_count = max(n_dim // 4, 2)
            d_count = max(n_dim // 2, 4)
            n_fp16_args = a_count + b_count

            half = ir.HalfType()
            v2f16 = ir.VectorType(half, 2)
            output_ty = ir.LiteralStructType([f32] * d_count)
            func_ty = ir.FunctionType(output_ty,
                                      [v2f16] * n_fp16_args + [f32] * d_count, False)
            func = ir.Function(self.llvm_module, func_ty, name=name)
            entry = func.append_basic_block("entry")
            b = ir.IRBuilder(entry)

            idx0 = ir.Constant(i32, 0)
            idx1 = ir.Constant(i32, 1)

            def ext(v2, idx, nm):
                h = b.extract_element(v2, idx, f"{nm}_h")
                return b.fpext(h, f32, nm)

            # Extract A fragment elements: a0..a3 = args[0..3], each v2f16
            a_lo = [ext(func.args[j], idx0, f"a{j}_lo") for j in range(4)]
            a_hi = [ext(func.args[j], idx1, f"a{j}_hi") for j in range(4)]

            # Extract B fragment elements: b0..b_{b_count-1} = args[4..4+b_count-1]
            b_lo = [ext(func.args[4 + j], idx0, f"b{j}_lo") for j in range(b_count)]
            b_hi = [ext(func.args[4 + j], idx1, f"b{j}_hi") for j in range(b_count)]

            # C accumulators: args[n_fp16_args .. n_fp16_args + d_count - 1]
            c_args = [func.args[n_fp16_args + j] for j in range(d_count)]

            # Compute D registers: N/8 column pairs, 4 D regs per pair
            n_col_pairs = max(n_dim // 8, 1)
            d_vals = [None] * d_count

            for cp in range(n_col_pairs):
                bi0 = 2 * cp       # first B reg index for this column pair
                bi1 = 2 * cp + 1   # second B reg index

                # D[4cp+0]: even col, rows 0-7 (a0/a2 fragment)
                di = 4 * cp
                t0 = b.fmul(a_lo[0], b_lo[bi0], f"d{di}_p0")
                t1 = b.fmul(a_hi[0], b_hi[bi0], f"d{di}_p1")
                t2 = b.fmul(a_lo[2], b_lo[bi1], f"d{di}_p2")
                t3 = b.fmul(a_hi[2], b_hi[bi1], f"d{di}_p3")
                d = b.fadd(c_args[di], t0, f"d{di}_a0")
                d = b.fadd(d, t1, f"d{di}_a1")
                d = b.fadd(d, t2, f"d{di}_a2")
                d = b.fadd(d, t3, f"d{di}_a3")
                d_vals[di] = d

                # D[4cp+1]: odd col, rows 0-7 -> pass through C (lane 0)
                d_vals[4 * cp + 1] = c_args[4 * cp + 1]

                # D[4cp+2]: even col, rows 8-15 (a1/a3 fragment)
                di2 = 4 * cp + 2
                t0 = b.fmul(a_lo[1], b_lo[bi0], f"d{di2}_p0")
                t1 = b.fmul(a_hi[1], b_hi[bi0], f"d{di2}_p1")
                t2 = b.fmul(a_lo[3], b_lo[bi1], f"d{di2}_p2")
                t3 = b.fmul(a_hi[3], b_hi[bi1], f"d{di2}_p3")
                d = b.fadd(c_args[di2], t0, f"d{di2}_a0")
                d = b.fadd(d, t1, f"d{di2}_a1")
                d = b.fadd(d, t2, f"d{di2}_a2")
                d = b.fadd(d, t3, f"d{di2}_a3")
                d_vals[di2] = d

                # D[4cp+3]: odd col, rows 8-15 -> pass through C (lane 0)
                d_vals[4 * cp + 3] = c_args[4 * cp + 3]

            out = ir.Constant(output_ty, None)
            for j in range(d_count):
                out = b.insert_value(out, d_vals[j], j, f"wgmma_d{j}")
            b.ret(out)
        elif name == "hadd2":
            func_ty = ir.FunctionType(i32, [i32, i32], False)
            func = ir.Function(self.llvm_module, func_ty, "hadd2")
        elif name in ("hfma2", "hfma2_ftz"):
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, name)
        elif name in ("hmin2", "hmin2_ftz"):
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, name)
        elif name in ("hmax2", "hmax2_ftz"):
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, name)
        elif name == "barrier":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            func = self.ir.Function(self.llvm_module, func_ty, "syncthreads")
        elif name in ("named_barrier_sync", "named_barrier_arrive"):
            # Partial/warpgroup named barrier (BAR.SYNC/BAR.ARV <id>, <count>).
            # void named_barrier_{sync,arrive}(i32 id, i32 count). The wrapper
            # synchronizes exactly `count` participants at named barrier `id`
            # (fiber yield on the warp backend; NOP on per-thread x86).
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_rs_a_spill":
            # RS-form WGMMA A-fragment spill to per-warpgroup scratch (FA PV GEMM,
            # A=P in registers). void(i32 r0, r1, r2, r3) — the 4 f16x2 A regs.
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_rs_a_read":
            # Read A[m,k] (fp16 bits, zero-extended) from the spilled warpgroup
            # scratch. i32(i32 m, i32 k).
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_rs_a_spill_fp8":
            # FP8 (e4m3) RS-form WGMMA A-fragment spill (FA3 FP8 PV GEMM). The 4
            # i32 regs each hold 4 e4m3 bytes (K-step 32). void(i32 r0,r1,r2,r3).
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_rs_a_read_fp8":
            # Read the e4m3 A byte (zero-extended in an i32) at (m,k) from the fp8
            # spilled warpgroup scratch (k in [0,32)). i32(i32 m, i32 k).
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_wgmma_dbg":
            # Diagnostic: void(i32 a_base, i32 b_base, i32 is_rs, i32 n_dim).
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_pv_boff_dbg":
            # Diagnostic: void(i32 n_val, i32 k_local, i32 b_off) — dump the PV B-read
            # smem byte offset to correlate vs the V-transpose STSM dst offsets.
            i32 = self.ir.IntType(32)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_s_dump":
            # Diagnostic: void(i32 m, i32 n, i32 a_base, i32 b_base, float v) — dump a
            # QK S fragment + its atom (a_base) AND Q-stage (b_base) discriminators for
            # element-wise compare vs a CPU reference. Gated in the wrapper.
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_dqk_dump":
            # Diagnostic: void(i32 n_val, i32 kc, i32 b_base, i32 b_kg0, i32 b_off,
            # float bv) — dump the hd96 dQ K-read (dq_kread) for KID-injected noise-free
            # decode of read_hd vs intended n_val. Gated SLIFTER_KID_INJECT.
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32, i32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_pvacc_dump":
            # Diagnostic: void(i32 m, i32 n, float v) — dump the PV accumulator.
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_ex2_dump":
            # Diagnostic: void(float in, float out) — dump the bwd softmax EX2
            # input (S*scale - lse) and output (P) to pin why P reads 1.0.
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_pvop_dump":
            # Diagnostic: void(i32 m, i32 n, i32 k_local, i32 b_base, float a, float b)
            # — dump the PV (RS-form) per-k A=P and B=V operands + b_base (sub-block
            # discriminator) for alignment check vs CPU P and V. Gated in the wrapper.
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32, f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_sop_dump":
            # Diagnostic: void(i32 m, i32 n, i32 k_local, i32 a_base, i32 b_base,
            # float av, float bv) — dump the SS-form (S/dP/dQ) per-k A/B operands +
            # both bases to pin swapAB SS base resolution vs data presence. Gated.
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32, i32, i32, i32, f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_redg_dump":
            # Diagnostic: void(i64 addr, float val) — dump the F32x{2,4} REDG
            # dQ_accum atomic-add (gmem byte address + value) so the dQ_accum gmem
            # layout (C-fragment -> gmem) can be decoded vs the reference. Gated.
            i64 = self.ir.IntType(64)
            f32 = self.ir.FloatType()
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_ostore_dbg":
            # Diagnostic: void(i64 addr, i32 width) — print per-lane O-store address
            # to decode the packgqa O-store column mapping. Gated in the wrapper.
            i32 = self.ir.IntType(32)
            i64 = self.ir.IntType(64)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_ldgsts_dbg":
            # Diagnostic: void(i64 global_addr, i32 shared_off) — per-thread
            # LDGSTS gather: which global byte address lands in which shared slot.
            # Decodes the packgqa Q-gather M-row->query mapping. Gated in wrapper.
            i32 = self.ir.IntType(32)
            i64 = self.ir.IntType(64)
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "utma_load":
            # TMA tensor bulk copy (UTMALDG) lowered to one runtime call into the
            # x86 wrapper. The wrapper reads the host-materialized SlifterTmaDesc
            # at desc_addr (global base, rows, cols_k, elem_bytes, swizzle),
            # applies the 4 recovered 4D tensor coords {c0,c1,c2,c3} (UR10..UR13
            # window, recovered by TmaCoordRecover; all 0 for a 2D single tile),
            # and writes the box-sized tile into shared memory at
            # smem_base+smem_off using the descriptor's smem layout/swizzle.
            #   void utma_load(i64 desc_addr, i8* smem_base, i32 smem_off,
            #                  i32 c0, i32 c1, i32 c2, i32 c3)
            i8p = self.ir.PointerType(i8)
            func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                           [i64, i8p, i32, i32, i32, i32, i32],
                                           False)
            func = self.ir.Function(self.llvm_module, func_ty, "utma_load")
        elif name == "utma_store":
            # TMA bulk STORE (UTMASTG), the inverse of utma_load: copies the O
            # tile from shared memory out to the global O tensor through the
            # host-materialized SlifterTmaDesc. Same 7-arg signature.
            #   void utma_store(i64 desc_addr, i8* smem_base, i32 smem_off,
            #                   i32 c0, i32 c1, i32 c2, i32 c3)
            i8p = self.ir.PointerType(i8)
            func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                           [i64, i8p, i32, i32, i32, i32, i32],
                                           False)
            func = self.ir.Function(self.llvm_module, func_ty, "utma_store")
        elif name == "ublk_reduce_add":
            # TMA bulk REDUCE-add (UBLKRED.G.S.ADD.F32): gmem[addr .. addr+size]
            # += smem[smem_base+smem_off .. +size], element-wise fp32. The FA3
            # backward dQ_accum accumulation. Lowered to one runtime call; the
            # element loop + atomic add (for CTA concurrency) live in the wrapper.
            #   void ublk_reduce_add(i64 gmem_addr, i8* smem_base, i32 smem_off,
            #                        i64 size_bytes)
            i8p = self.ir.PointerType(i8)
            func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                           [i64, i8p, i32, i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "ublk_reduce_add")
        elif name in ("warp_region_push", "warp_region_pop"):
            # Divergence-history events from SASS's own reconvergence markers
            # (BSSY/BSYNC). Shadow-mode
            # step 1: runtime records per-lane history; collectives compare.
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name in ("warp_turn_begin", "warp_turn_end"):
            # Lockstep token turnstile (CULIFTER_LOCKSTEP=1): brackets each
            # shared-memory access so pre-Volta warp-synchronous code sees
            # SIMT round ordering. No-ops unless SLIFTER_LOCKSTEP=1 at run.
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "barrier0_and":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "barrier0_and")
        elif name == "barrier0_or":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "barrier0_or")
        elif name == "barrier0_popc":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "barrier0_popc")
        elif name.startswith("vote_"):
            # C++ `bool` parameters are i8 in the x86_64 SysV ABI (and clang emits
            # i8 for `bool` arguments). Declaring the IR parameter as `i1` causes
            # an ABI mismatch at -O2: LLVM elides the `and $1, %esi` mask before
            # the call, so the caller's full 32-bit predicate value spills into
            # SIL, and the receiver's `testb %sil, %sil` (== `bool pred != 0`)
            # sees the high-byte garbage instead of bit 0. The result is `pred`
            # appearing as a non-zero digit for every lane, ballot=0xFFFFFFFF
            # everywhere, and the cub Onesweep MatchAny `~ballot` side collapsing
            # to 0 (the cub_partial / sa-cuda partial-tile bug). Declaring i8 and
            # letting run_intrinsic auto-zext i1->i8 forces LLVM to honor the
            # ABI; the C++ wrapper still receives a clean 0/1.
            vote_ret_map = {"vote_any": i8, "vote_all": i8, "vote_uni": i8, "vote_ballot": i32}
            if name in vote_ret_map:
                # Trailing i32 = collective site id (SASS address); see
                # CpuLifter.collective_site_arg.
                func_ty = self.ir.FunctionType(vote_ret_map[name], [i32, i8, i32], False)
                func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name.startswith("shfl_"):
            parts = name.split("_")
            mode, dtype_name = parts[1], parts[2]
            dtype_map = {"i32": i32, "i64": i64, "f32": f32, "f64": f64}
            dtype = dtype_map.get(dtype_name)
            if dtype:
                func_ty = self.ir.FunctionType(dtype, [i32, dtype, i32, i32, i32], False)
                func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name.startswith("match_any_"):
            dtype_name = name.split("_")[-1]
            dtype = {"i32": i32, "i64": i64}.get(dtype_name)
            if dtype:
                func_ty = self.ir.FunctionType(i32, [i32, dtype, i32], False)
                func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name.startswith("redux_"):
            # warp-wide reduction (kernel_wrapper_75.h warp_redux family)
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "elect_one":
            # ELECT leader election: (site) -> leader lane id (kernel_wrapper_75.h)
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name in ("slifter_mbar_init", "slifter_mbar_arrive",
                      "slifter_mbar_complete_tx"):
            # Hopper mbarrier runtime (kernel_wrapper_75.h). init(off,i64),
            # arrive(off,i64 tx), complete_tx(off,i64 bytes) -> void.
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_mbar_arrive_tx":
            # Byte-accounting expect-tx arrive: arrive_tx(off, bytes) -> void. Records
            # the barrier + the transaction bytes it expects; the flip is deferred
            # until those bytes are delivered by the bulk copies (kernel_wrapper_75.h).
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name in ("slifter_mbar_arrive_deferred", "slifter_mbar_ldgsts_arrive"):
            # arrive_deferred(off) -> void: count-arrive whose phase flip is deferred
            # to the producer fiber's next yield (fp8 Vt-transpose-done arrive).
            # ldgsts_arrive(off) -> void: cp.async transaction-completion arrive
            # (ARRIVES.LDGSTSBAR.TRANSCNT) — a release fence + NO count (the paired
            # SYNCS.ARRIVE.A1T0 supplies the count). Both in kernel_wrapper_75.h.
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_mbar_try_wait":
            # try_wait(off, parity) -> i32 ready (1 = barrier advanced past parity)
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_ublkcp_dump":
            # DEBUG (gated): dump(dst_off:i32, src:i8*, bytes:i64) -> void
            func_ty = self.ir.FunctionType(
                self.ir.VoidType(),
                [i32, self.ir.PointerType(self.ir.IntType(8)), i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_lds_dump":
            # DEBUG (gated): dump(eff_off:i32, imm_off:i32) -> void
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_sts_dump":
            # DEBUG (gated SLIFTER_LSE_TRACE): sts_dump(eff_off:i32, val:i8*, bytes:i32)
            func_ty = self.ir.FunctionType(
                self.ir.VoidType(),
                [i32, self.ir.PointerType(self.ir.IntType(8)), i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_dqacc_sts_dump":
            # DEBUG (gated SLIFTER_DQACC_STS): dqacc_sts_dump(eff_off:i32, vals:i8*,
            # nregs:i32) -> void.  Records the per-thread dQ_accum vector STS so the
            # (smem_flat -> query,hdim) decode map can be recovered offline.
            func_ty = self.ir.FunctionType(
                self.ir.VoidType(),
                [i32, self.ir.PointerType(self.ir.IntType(8)), i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "slifter_mbar_deliver_tx":
            # byte-accounting delivery: deliver_tx(bytes) -> void (retires `bytes`
            # against the oldest outstanding expect-tx barrier; flips it when full).
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, name)
        elif name == "brev32":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.bitreverse.i32")
        elif name == "brev64":
            func_ty = self.ir.FunctionType(i64, [i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.bitreverse.i64")
        elif name == "trap":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.trap")
        elif name == "prmt":
            func_ty = self.ir.FunctionType(i32, [i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "prmt")
        elif name == "ctlz_i32":
            func_ty = self.ir.FunctionType(i32, [i32, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctlz.i32")
        elif name == "ctlz_i64":
            func_ty = self.ir.FunctionType(i32, [i64, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctlz.i64")
        elif name == "ctpop_i32":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctpop.i32")
        elif name.startswith("isspacep_"):
            space_name = name.split("_")[1]  # const, global, local, shared
            func_name = f"is_{space_name}_address"
            func_ty = self.ir.FunctionType(i1, [i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, func_name)
        elif name in ("fma_rm_f32", "fma_rp_f32", "fma_rz_f32"):
            # SASS FFMA.RM/.RP/.RZ on the CPU backend: x86 does not expose the
            # NVPTX directed-rounding FMAs. ptxas's exp2 / sigmoid lowering uses
            # them with the 2^23+1 BIAS-trick to round x*c to a representable
            # integer; for the inputs hit by HeCBench element-wise kernels the
            # difference between round-to-nearest-even and the SASS directed
            # mode is sub-ULP, so emitting llvm.fma.f32 keeps the value finite
            # and lets sigmoid/tanh produce correct results. Without this entry
            # the lifter falls back to a zero-fill placeholder, which silently
            # propagates 0 into subsequent FMAs and yields NaN at the output.
            func = self.llvm_module.globals.get("llvm.fma.f32")
            if func is None:
                func_ty = self.ir.FunctionType(f32, [f32, f32, f32], False)
                func = self.ir.Function(self.llvm_module, func_ty, "llvm.fma.f32")
        elif name == "rcp_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "rcp_f32")
        elif name == "rcp_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "rcp_f64")
        elif name == "rsqrt_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "rsqrt_f32")
        elif name == "rsqrt_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "rsqrt_f64")
        elif name == "sqrt_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.sqrt.f32")
        elif name == "sin_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.sin.f32")
        elif name == "cos_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.cos.f32")
        elif name == "exp2_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.exp2.f32")
        elif name == "log2_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.log2.f32")
        elif name == "floor_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.floor.f32")
        elif name == "floor_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.floor.f64")
        elif name == "ceil_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ceil.f32")
        elif name == "ceil_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ceil.f64")
        elif name == "trunc_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.trunc.f32")
        elif name == "trunc_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.trunc.f64")
        elif name == "rint_f32":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.rint.f32")
        elif name == "rint_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.rint.f64")
        
        if func is None:
            raise UnsupportedInstructionException(f"Unknown intrinsic {name}")
        
        self.intrinsics[name] = func
        return func

    def run_intrinsic(self, name, builder, args, call_name=""):
        # fchk checks if fp division neds to be done in slow path
        # implementation unknown, always return false
        if name == "fchk":
            return ir.Constant(ir.IntType(1), 0)
        
        
        func = self._get_intrinsic(name)
        
        
        if name == "abs":
            # llvm.abs.i32 needs (value, is_int_min_poison)
            args = list(args) + [ir.Constant(ir.IntType(1), 0)]
        elif name in ("ctlz_i32", "ctlz_i64"):
            # llvm.ctlz needs is_zero_undef
            args = list(args) + [ir.Constant(ir.IntType(1), 0)]
        
        # Cast args to match function parameter types
        func_arg_types = func.type.pointee.args
        coerced = []
        for i, (a, expected_ty) in enumerate(zip(args, func_arg_types)):
            if a.type != expected_ty:
                a = self._insert_type_cast_if_needed(builder, a, expected_ty, f"{call_name}_arg{i}")
            coerced.append(a)
        return builder.call(func, coerced, call_name)

    def _load_const_mem_u32(self, builder, offset, tag):
        ptr = builder.gep(
            self.ConstMem,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), offset),
            ],
            f"{tag}_ptr",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(self.ir.IntType(32)))
        return builder.load(ptr, tag)

    def _load_lane_id_value(self, builder):
        offset = SR_TO_OFFSET.get("SR_LANEID")
        if offset is None:
            return self.ir.Constant(self.ir.IntType(32), 0)
        return self._load_const_mem_u32(builder, offset, "laneid")

    def _emit_lane_id_i32(self, builder):
        # Backend hook used by shared lifters (e.g. lift_shfl's destination
        # predicate). The wrapper writes each thread's lane id into its
        # thread_local const_mem at SR_LANEID (0x38 on every supported ISA).
        return self._load_lane_id_value(builder)

    def _emit_lane_mask(self, builder, kind, tag):
        lane_i32 = self._load_lane_id_value(builder)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        lane_i64 = builder.zext(lane_i32, i64, f"{tag}_lane")
        one64 = self.ir.Constant(i64, 1)
        all_ones64 = self.ir.Constant(i64, 0xFFFFFFFFFFFFFFFF)

        if kind == "eq":
            mask64 = builder.shl(one64, lane_i64, f"{tag}_shift")
        elif kind == "le":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.sub(
                builder.shl(one64, lane_inc, f"{tag}_shift"), one64, f"{tag}_sub"
            )
        elif kind == "lt":
            mask64 = builder.sub(
                builder.shl(one64, lane_i64, f"{tag}_shift"), one64, f"{tag}_sub"
            )
        elif kind == "ge":
            mask64 = builder.shl(all_ones64, lane_i64, f"{tag}_shift")
        elif kind == "gt":
            lane_inc = builder.add(lane_i64, one64, f"{tag}_inc")
            mask64 = builder.shl(all_ones64, lane_inc, f"{tag}_shift")
        else:
            mask64 = self.ir.Constant(i64, 0)

        return builder.trunc(mask64, i32, tag)

    # ------------------------------------------------------------------
    # Vector load/store helpers for .128 / .64 wide memory operations
    # ------------------------------------------------------------------

    def _vec_set_val(self, IRBuilder, IRRegs, dest, vec_val):
        """Extract i32 elements from a vector value and assign to reg_group sub-registers.

        dest   — Operand with is_reg_group=True (2 or 4 sub-registers)
        vec_val — llvmlite IR value of type <N x i32>
        """
        i32 = self.ir.IntType(32)
        name = dest.name or "reg_group"
        for i, sub_op in enumerate(dest.reg_group):
            elem = IRBuilder.extract_element(
                vec_val, self.ir.Constant(i32, i), f"{name}_v{i}"
            )
            IRRegs[sub_op.get_ir_name(self)] = elem

    def _vec_get_val(self, IRBuilder, IRRegs, val_op, name=""):
        """Build a <N x i32> vector from a reg_group operand's sub-registers.

        val_op — Operand with is_reg_group=True (2 or 4 sub-registers)
        Returns an IR value of type <N x i32>.
        """
        i32 = self.ir.IntType(32)
        reg_count = len(val_op.reg_group)
        vec_ty = self.ir.VectorType(i32, reg_count)
        vec = self.ir.Constant(vec_ty, self.ir.Undefined)
        for i, sub_op in enumerate(val_op.reg_group):
            sub_val = self._get_val(IRBuilder, IRRegs, sub_op,
                                    f"{name}_r{i}" if name else f"vec_r{i}")
            sub_val = self._insert_type_cast_if_needed(
                IRBuilder, sub_val, i32,
                f"{name}_r{i}_i32" if name else f"vec_r{i}_i32"
            )
            vec = IRBuilder.insert_element(
                vec, sub_val, self.ir.Constant(i32, i),
                f"{name}_ins{i}" if name else f"vec_ins{i}"
            )
        return vec

    def _get_val(self, builder, ir_regs, op, name=""):
        # For const_mem with RZ offset (e.g. c[0x0][RZ]), treat as load from offset 0,
        # not as the zero constant.
        if op.is_const_mem:
            if op.is_rz:
                byte_offset = int(op.offset_value) if op.offset_value is not None else 0
                target_ty = op.get_ir_type(self)
                return self._load_from_const_mem(builder, op.const_mem_bank, byte_offset, target_ty, name)
            val = self._get_const_mem_val(builder, ir_regs, op, name)

        elif op.is_rz:
            return self.ir.Constant(op.get_ir_type(self), 0)

        elif op.is_pt:
            return self.ir.Constant(op.get_ir_type(self), int(not op.is_not_reg))

        elif op.is_immediate:
            ty = op.get_ir_type(self)
            val = op.immediate_value
            # Float immediate stored in integer type: convert to bit pattern
            if isinstance(val, float) and hasattr(ty, 'width') and ty.intrinsic_name.startswith('i'):
                import struct
                if ty.width == 32:
                    val = struct.unpack('<I', struct.pack('<f', val))[0]
                elif ty.width == 64:
                    val = struct.unpack('<Q', struct.pack('<d', val))[0]
            # Integer immediate used in float context (e.g. MOV R9, 0x437c0000
            # where R9 is later used by FFMA): bitcast the int bits to float,
            # not convert int value to float.
            elif isinstance(val, int) and not isinstance(val, bool) and hasattr(ty, 'intrinsic_name') and ty.intrinsic_name == 'f32':
                import struct
                val = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
            elif isinstance(val, int) and not isinstance(val, bool) and hasattr(ty, 'intrinsic_name') and ty.intrinsic_name == 'f64':
                import struct
                val = struct.unpack('<d', struct.pack('<Q', val & 0xFFFFFFFFFFFFFFFF))[0]
            # Bool immediate stored in integer type: convert to int
            if isinstance(val, bool):
                val = int(val)
            return self.ir.Constant(ty, val)

        elif op.is_mem_addr and not op.is_reg:  # E.g. LDS.U R2 = [0xc]
            # The address offset for absolute mem-addr operands lives in
            # `offset_value`, not `immediate_value` (which is only populated
            # for is_immediate operands). Using immediate_value here returns
            # None and llvmlite silently generates Constant(0) — making
            # `LDS R4, [0x4]` indistinguishable from `LDS R4, [0]`.
            return self.ir.Constant(op.get_ir_type(self), op.offset_value or 0)

        elif op.is_reg_group:
            reg_count = len(op.reg_group)
            total_bits = reg_count * 32
            val = self.ir.Constant(self.ir.IntType(total_bits), 0)
            for i, sub_op in enumerate(op.reg_group):
                sub_val = self._get_val(
                    builder, ir_regs, sub_op,
                    f"{name}_r{i}" if name else f"reg_group_r{i}",
                )
                sub_val_i32 = self._insert_type_cast_if_needed(
                    builder,
                    sub_val,
                    self.ir.IntType(32),
                    f"{name}_r{i}_i32" if name else f"reg_group_r{i}_i32",
                )
                sub_val_ext = builder.zext(
                    sub_val_i32, self.ir.IntType(total_bits),
                    f"{name}_r{i}_ext" if name else f"reg_group_r{i}_ext",
                )
                if i > 0:
                    sub_val_ext = builder.shl(
                        sub_val_ext,
                        self.ir.Constant(self.ir.IntType(total_bits), i * 32),
                        f"{name}_r{i}_shift" if name else f"reg_group_r{i}_shift",
                    )
                val = builder.or_(
                    val, sub_val_ext,
                    f"{name}_packed_{i}" if name else f"reg_group_packed_{i}",
                )

        elif op.is_reg:
            key = op.get_ir_name(self)
            if key in ir_regs:
                val = ir_regs[key]
            else:
                # Register is used but not defined in ir_regs. Most commonly
                # this happens after HandleCall inlining clones blocks whose
                # register uses refer to a definition outside the cloned
                # scope (SSA renaming gap). Matches the phi-node fallback
                # at line 441: treat as zero of the expected type.
                #
                # Under CULIFTER_NO_FALLBACK (ablation B3/B4), raise instead
                # so the ablation observes actual lift failure rather than
                # silent zero-substitution.
                import os as _os
                if _os.environ.get("CULIFTER_NO_FALLBACK") == "1":
                    raise KeyError(
                        f"_get_val: undefined register {key} "
                        f"(strict mode, CULIFTER_NO_FALLBACK=1)"
                    )
                target_ty = op.get_ir_type(self)
                if self._verbose:
                    print(f"[lift] _get_val: {key} unresolved — using zero "
                          f"of {target_ty} (likely SSA/inline interaction)")
                val = self.ir.Constant(target_ty, 0)
        else:
            raise UnsupportedInstructionException(f"Unsupported operand type: {op}")

        # `.SIGN` operand modifier (PLOP3/LOP3 overflow-test idiom): the LUT
        # consumes the SIGN BIT (bit 31) of the register, not the whole value.
        # These operands are typed Bool (i1) by TypeAnalysis, so the default
        # i32->i1 coercion below would `trunc` and grab bit 0 (wrong). Extract
        # bit 31 here first: sign-bit -> i1. (CUTLASS .satfinite IMMA emits
        # `PLOP3.LUT P, PT, Ra.SIGN, Rb.SIGN, Rc.SIGN, ...` add-overflow tests;
        # mis-reading bit 0 makes the saturation predicate misfire -> spurious
        # INT_MIN/INT_MAX clamps.)
        if getattr(op, "suffix", None) == "SIGN" and isinstance(val.type, self.ir.IntType):
            if val.type.width > 1:
                signbit = builder.lshr(
                    val, self.ir.Constant(val.type, val.type.width - 1),
                    f"{name}_signshift" if name else "signshift")
                val = builder.trunc(signbit, self.ir.IntType(1),
                                    f"{name}_signbit" if name else "signbit")

        target_ty = op.get_ir_type(self)
        val = self._insert_type_cast_if_needed(builder, val, target_ty, name)

        type_desc = op.get_type_desc()
        if op.is_negative_reg:
            if type_desc.startswith("F"):
                val = builder.fneg(val, f"{name}_fneg")
            else:
                val = builder.neg(val, f"{name}_neg")
        if op.is_not_reg:
            val = builder.not_(val, f"{name}_not")
        if op.is_abs_reg:
            if type_desc in FLOAT64:
                val = self.run_intrinsic("fabs_f64", builder, [val], f"{name}_fabs")
            elif type_desc in FLOAT32:
                val = self.run_intrinsic("fabs", builder, [val], f"{name}_fabs")
            else:
                val = self.run_intrinsic("abs", builder, [val], f"{name}_abs")

        if op.is_mem_addr and op.offset:
            val = builder.add(
                val, self.ir.Constant(val.type, op.offset), f"{name}_memaddr_offset_add"
            )

        return val

    def _as_pointer(self, builder, addr_val, pointee_ty, name):
        if addr_val.type.is_pointer:
            return addr_val

        if addr_val.type.intrinsic_name[0] != 'i':
            raise InvalidTypeException(
                f"Expected integer address for {name}, got {addr_val.type}"
            )

        if addr_val.type.width < 64:
            addr_val = builder.zext(addr_val, self.ir.IntType(64), f"{name}_zext64")
        elif addr_val.type.width > 64:
            addr_val = builder.trunc(addr_val, self.ir.IntType(64), f"{name}_trunc64")

        ptr_ty = self.ir.PointerType(pointee_ty)
        return builder.inttoptr(addr_val, ptr_ty, name)

    def _lift_function(self, func, llvm_module):
        self._setup_const_mem(llvm_module, func.name)

        # Per-function descriptor const map (RecordDescriptorConsts): consulted by
        # _resolve_descriptor_addr when a desc[URx] base is absent from IRRegs
        # (its defining ULDC.64 was renamed/DCE'd).
        self._desc_const_map = getattr(func, "desc_const_map", {})

        # Per-function FP8 (e4m3/e5m2) V-transpose detection. The in-kernel V
        # transpose (LDSM.16.MT88 + PRMT + STSM.16.M88) runs BEFORE the QGMMA
        # that carries the fp8 token, and the LDSM/STSM opcodes themselves are
        # always .16/b16, so the 1-byte element granularity can't be read off
        # them. Detect the fp8 transpose by its unique fingerprint: a PRMT whose
        # selector is the even/odd byte de-interleave 0x6420 or 0x7531. That
        # idiom only appears to complete an 8-bit (e4m3/e5m2) transpose; fp16/
        # bf16 kernels also use STSM.16.M88 (for the O epilogue) but never these
        # selectors, so they stay byte-identical on the b16 transpose path.
        # (An E4M3/E5M2 GMMA opcode is also accepted as a belt-and-suspenders
        # signal, e.g. for variants that transpose without that exact PRMT.)
        def _is_fp8_deinterleave(inst):
            if inst.opcodes and inst.opcodes[0] in ("PRMT", "UPRMT"):
                for op in inst.get_uses():
                    if getattr(op, "is_immediate", False) and \
                       op.immediate_value in (0x6420, 0x7531):
                        return True
            return False
        self._fa_fp8 = any(
            ("E4M3" in inst.opcodes or "E5M2" in inst.opcodes
             or _is_fp8_deinterleave(inst))
            for bb in func.blocks for inst in bb.instructions
        )

        # Per-function FlashAttention-backward detection. The bwd mainloop is the
        # only place a WGMMA descriptor carries `.tnspA` (the SdP/dQ GEMMs are
        # SwapAB so ptxas spells the physically-swapped A descriptor MN-major).
        # The fwd kernels NEVER emit `.tnspA` (verified across all fwd cubins), so
        # this flag stays False for every fwd kernel — leaving the fwd RS-form V
        # (`.tnspB`, MN-major in smem) on its existing is_mn_b path byte-identical.
        # It is needed because the RS-form dKV GEMMs (dV=Pᵀ·dO, dK=dSᵀ·Q) carry
        # `.tnspB` too but read a K-major Q/dO tile transposed with an MN_ATOM_STRIDE
        # of kBlockM*64 (queries) — distinct from the fwd PV's kBlockN*64 (seqlen).
        # Those RS GEMMs precede the `.tnspA` dQ in instruction order, so the
        # discriminator must be a whole-function pre-scan, not per-instruction state.
        # Also capture kBlockM = the N dim of the bwd SS GEMMs (= the `.tnspA` GEMM's
        # N), used as the bwd dKV B-read MN atom stride.
        self._fa_bwd = False
        self._fa_kblockm = None
        # Rolling "N of the most recent non-bwd SS-form GEMM" — see the capture
        # site in _wgmma_sw128_product for its exact (order-dependent) semantics.
        # Reset per function so a value can never leak from a previously lifted
        # function in the same module (before this reset it was set lazily and
        # survived across functions).
        self._fa_kblockn = None
        # The SS-form dKV GEMM's N dimension = kBlockN (keys). This GEMM is the
        # one carrying BOTH `.tnspA` AND `.tnspB` (dKV_swapAB=1 SS-form, e.g.
        # hdim192's HGMMA.64x96x16.tnspA.tnspB). It does NOT exist in the RS-form
        # dKV class (hdim64/128, Mma_dKV_is_RS=1, dKV `.tnspB`-only), where the bwd
        # MN_ATOM_STRIDE path is never taken (the RS dKV uses kBlockM*64 via
        # bwd_dkv_b instead), so `_fa_kblockn_bwd` stays None there and those
        # kernels are byte-identical. Capturing it here lets the SS-form dKV B-read
        # (is_mn_b, MN_ATOM_STRIDE) use the real kBlockN instead of the hard-coded
        # fwd fallback 176 (which overshoots the dS tile -> smem-guard crash).
        self._fa_kblockn_bwd = None
        # The SmemLayoutPdS atom K-extent = kBlockN/(NumMmaWarpGroups/AtomLayoutMSdP)
        # = the per-WG key count. This equals the N dimension of the plain (no-tnsp)
        # SS GEMMs (QK/SdP/dP: TileShapeAtomSdP N = kBlockN/(NumWG/AtomLayoutMSdP)).
        # It selects the PdS swizzle for the dQ A-read (dq_pds_a): atom 64 ->
        # Sw<3,4,3> (hd64/softcap class), atom 32 -> Sw<2,4,3> (hd192). Captured
        # from the smallest plain bwd HGMMA N. None for fwd (no UBLKRED gate below).
        self._fa_pds_atom = None
        # UBLKRED.G.S.ADD (dQ_accum bulk-reduce-add) is emitted ONLY by the FA
        # backward mainloop (the dQ_accum store); every fwd kernel has zero. It is
        # the dtype/geometry-independent bwd discriminator used by the dQ_swapAB=0
        # dS A-read fix below (the hdim64/softcap class has NO `.tnspA`, so the
        # `_fa_bwd` flag — keyed on `.tnspA` — is False there and cannot gate it).
        self._has_ublkred = False
        # The bwd dQ GEMM (dQ = dS·K) contracts over the keys = kBlockN, so its
        # accumulating-HGMMA count × k_dim == kBlockN. The dQ GEMM is the tnspB-only
        # HGMMA with the SMALLEST N (= kHeadDim/(NumWG/AtomLayoutMdQ); the larger
        # tnspB-only group is the dP GEMM, N = full hdim). kBlockN is needed for the
        # SmemLayoutKt hd-atom stride of the dq_kread B-read (hd96), where it is NOT
        # any GEMM's N and `_fa_kblockn` is unreliable (the bwd has no `.tnspA`, so the
        # fwd capture overwrites it with the dQ GEMM N=48). Noise-free
        # KID-injection established that the hd-atom stride is
        # kBlockN*ATOM = 128*16 = 2048; the old `_fa_kblockn`(=48)*16=768 double-
        # under-counted. Tally (N -> accum-count, k_dim) over tnspB-only HGMMAs.
        _dqk_naccum = {}   # N -> [count, k_dim]
        for bb in func.blocks:
            for inst in bb.instructions:
                if inst.opcodes and inst.opcodes[0] == "UBLKRED":
                    self._has_ublkred = True
                if not (inst.opcodes and inst.opcodes[0] in ("HGMMA", "WGMMA", "QGMMA")):
                    continue
                _has_tnsa = False
                _has_tnsb = False
                for op in inst.get_uses():
                    nm = str(getattr(op, "name", "") or "")
                    suf = str(getattr(op, "suffix", "") or "")
                    if "tnspA" in nm or "tnspA" in suf:
                        _has_tnsa = True
                    if "tnspB" in nm or "tnspB" in suf:
                        _has_tnsb = True
                shp = inst.opcodes[1] if len(inst.opcodes) > 1 else ""
                sp = shp.split("x")
                _gemm_n = int(sp[1]) if len(sp) > 1 and sp[1].isdigit() else None
                # k_dim (contraction step) = the shape's 3rd field, stripped of any
                # ".F32"/dtype suffix (e.g. "16" from "64x48x16.F32").
                _kfield = sp[2].split(".")[0] if len(sp) > 2 else ""
                _gemm_k = int(_kfield) if _kfield.isdigit() else None
                if _has_tnsa:
                    self._fa_bwd = True
                    if _gemm_n is not None:
                        self._fa_kblockm = _gemm_n
                    # SS-form dKV GEMM (both `.tnspA` and `.tnspB`): N = kBlockN.
                    if _has_tnsb and _gemm_n is not None:
                        self._fa_kblockn_bwd = _gemm_n
                elif _has_tnsb and _gemm_n is not None and _gemm_k is not None:
                    # tnspB-only HGMMA: tally accumulating count per N (the dQ/dP GEMMs).
                    e = _dqk_naccum.setdefault(_gemm_n, [0, _gemm_k])
                    e[0] += 1
                elif (not _has_tnsb) and _gemm_n is not None:
                    # Plain (no-tnsp) SS GEMM: N == PdS atom K-extent. Track the
                    # smallest (the SdP/QK/dP atom N; ignores any larger plain GEMM).
                    if self._fa_pds_atom is None or _gemm_n < self._fa_pds_atom:
                        self._fa_pds_atom = _gemm_n

        # dQ GEMM kBlockN = (its accumulating-HGMMA count) × k_dim. The dQ GEMM is the
        # SMALLEST-N tnspB-only group (the larger is dP, N = full hdim). Used ONLY by
        # the hd96 dq_kread SmemLayoutKt hd-atom stride (gated swiz_B_b<3); None
        # everywhere else so non-dq_kread reads are byte-identical.
        self._fa_kblockn_dqk = None
        if _dqk_naccum:
            _min_n = min(_dqk_naccum)
            _cnt, _kd = _dqk_naccum[_min_n]
            self._fa_kblockn_dqk = _cnt * _kd

        func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
        ir_function = self.ir_functions.get(func.name)
        if ir_function is None:
            ir_function = self.ir.Function(llvm_module, func_ty, func.name)
            self.ir_functions[func.name] = ir_function

        func.block_map = {}
        func.build_bb_to_ir_map(ir_function, func.block_map)

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.block_map[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.block_map)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.block_map[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.block_map)

        self._ensure_block_terminators(func)

    def lift_instruction(self, Inst, IRBuilder: ir.IRBuilder, IRRegs, block_map):
        if len(Inst.opcodes) == 0:
            raise UnsupportedInstructionException("Empty opcode list")
        opcode = Inst.opcodes[0]

        _get_val = partial(self._get_val, IRBuilder, IRRegs)
        _as_pointer = partial(self._as_pointer, IRBuilder)

        handler = self.liftMap.get(opcode)
        if handler is None:
            raise UnsupportedInstructionException(f"Unhandled instruction: {opcode}")
        return handler(Inst, IRBuilder, IRRegs, block_map, _get_val=_get_val, _as_pointer=_as_pointer)
    
    @lift_for("S2R", "S2UR")
    def lift_s2r(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        valop = Inst.get_uses()[0]
        offset = SR_TO_OFFSET.get(valop.name)
        if offset is not None:
            val = self._load_const_mem_u32(IRBuilder, offset, f"sr_{offset:x}")
        elif valop.is_warp_size:
            val = self.ir.Constant(self.ir.IntType(32), 32)
        elif valop.is_active_mask:
            val = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
        elif valop.is_lane_mask_eq:
            val = self._emit_lane_mask(IRBuilder, "eq", "lanemask_eq")
        elif valop.is_lane_mask_le:
            val = self._emit_lane_mask(IRBuilder, "le", "lanemask_le")
        elif valop.is_lane_mask_lt:
            val = self._emit_lane_mask(IRBuilder, "lt", "lanemask_lt")
        elif valop.is_lane_mask_ge:
            val = self._emit_lane_mask(IRBuilder, "ge", "lanemask_ge")
        elif valop.is_lane_mask_gt:
            val = self._emit_lane_mask(IRBuilder, "gt", "lanemask_gt")
        elif valop.is_cluster_ctarank or valop.name.startswith("SR_SWINLO"):
            # SM90 cluster/scheduling registers — constant 0 on x86 (single CTA,
            # shared window based at 0). For SR_CgaCtaId this is exact, not an
            # approximation: the x86 wrapper models cluster dims (1,1,1), so
            # rank 0 keeps the (rank<<24)|offset LEA addressing inside the
            # CTA's own shared_mem buffer. The nvvm backend emits the real
            # %cluster_ctarank instead (lifter_nvvm lift_s2r/lift_s2ur).
            val = self.ir.Constant(self.ir.IntType(32), 0)
        elif valop.name.startswith("SR_SWINHI"):
            # High half of the generic shared-memory window. Kernels that
            # materialize generic pointers to __shared__ data (quicksort's
            # `__shared__ T *s`) build (SWINHI<<32 | 0x400+offset) and later
            # dereference with generic LD/ST. Emit a recognizable tag so
            # _redirect_shared_window can route those back into shared_mem.
            val = self.ir.Constant(self.ir.IntType(32), self.SHARED_WINDOW_TAG)
        else:
            print(f"S2R: Unknown special register {valop.name}")
            val = self.ir.Constant(self.ir.IntType(32), 0)
        self._set_val(IRBuilder, IRRegs, dest, val)


    def _resolve_descriptor_addr(self, Inst, ptr, addr, IRBuilder, IRRegs, _get_val):
        """Add descriptor base address for SM90 desc[URx][Ry.64] memory ops.

        The descriptor register (URx) holds a 64-bit base pointer loaded from
        const_mem. The register pair Ry.64 is an offset. The actual address
        is URx_base + Ry_offset.
        """
        desc_reg = getattr(ptr, 'desc_reg', None)
        if not desc_reg or desc_reg == 'URZ':
            return addr
        # Look up descriptor register value directly from IRRegs dict.
        # No exception guard here (one used to swallow everything and return
        # the un-based offset — a silently WRONG address, missing the
        # descriptor base, surfacing later as memory corruption). Any failure
        # below must crash with context. (The old guard never fired across the
        # FA3/bench corpus — verified 2026-07.)
        i64 = self.ir.IntType(64)
        i32 = self.ir.IntType(32)
        lo_val = IRRegs.get(desc_reg)
        if lo_val is None:
            # The descriptor-defining ULDC.64 was renamed by SSA/RegRemap
            # (desc_reg keeps the original parse-time name) and usually DCE'd
            # (liveness can't see the desc_reg-only use), so it's not in
            # IRRegs. Fall back to the const offset recorded pre-rename
            # (RecordDescriptorConsts) and load the 64-bit base straight from
            # const memory — needs neither the renamed reg nor the dead load.
            # This is the FA3 desc[UR14] cross-block descriptor resolution.
            dc = getattr(self, "_desc_const_map", None)
            ent = dc.get(desc_reg) if dc else None
            if ent is not None:
                bank, off = ent
                desc_64 = self._load_from_const_mem(
                    IRBuilder, bank, off, i64, "desc_base64")
                if addr.type.width < 64:
                    addr = IRBuilder.zext(addr, i64, "desc_off64")
                return IRBuilder.add(desc_64, addr, "desc_addr")
            return addr
        # Get the paired high register for 64-bit base
        reg_num = int(desc_reg.replace('UR', ''))
        hi_reg = f'UR{reg_num + 1}'
        hi_val = IRRegs.get(hi_reg)
        if hi_val is not None:
            lo_ext = IRBuilder.zext(lo_val, i64, "desc_lo64")
            hi_ext = IRBuilder.zext(hi_val, i64, "desc_hi64")
            hi_shifted = IRBuilder.shl(hi_ext, self.ir.Constant(i64, 32), "desc_hi_shl")
            desc_64 = IRBuilder.or_(lo_ext, hi_shifted, "desc_base64")
        else:
            desc_64 = IRBuilder.zext(lo_val, i64, "desc_base64")
        return IRBuilder.add(desc_64, addr, "desc_addr")

    def _redirect_shared_window(self, IRBuilder, addr, name):
        """Route generic addresses whose high half is SHARED_WINDOW_TAG (a
        lifted SR_SWINHI, see lift_s2r) into the shared_mem array. The low
        24 bits are the same 0x400-based byte index STS/LDS use (CgaCtaId
        is 0 on x86), so indexing shared_mem directly stays consistent."""
        if addr.type.is_pointer or addr.type.width != 64:
            return addr
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        hi64 = IRBuilder.lshr(addr, self.ir.Constant(i64, 32), f"{name}_swin_hi64")
        hi = IRBuilder.trunc(hi64, i32, f"{name}_swin_hi")
        is_sh = IRBuilder.icmp_unsigned(
            "==", hi, self.ir.Constant(i32, self.SHARED_WINDOW_TAG),
            f"{name}_is_shared")
        lo = IRBuilder.trunc(addr, i32, f"{name}_swin_lo")
        off = IRBuilder.and_(lo, self.ir.Constant(i32, 0xFFFFFF),
                             f"{name}_smem_off")
        sh_ptr = IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(i32, 0), off],
            f"{name}_smem_ptr")
        sh_addr = IRBuilder.ptrtoint(sh_ptr, i64, f"{name}_smem_addr")
        return IRBuilder.select(is_sh, sh_addr, addr, f"{name}_swin_addr")

    @lift_for("LDG", "LDG64", "LD")
    def lift_ldg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        # SM90 descriptor addressing: LDG.E desc[URx][Ry.64+offset]
        # descriptor_base + register_offset = actual address
        addr = _get_val(ptr, "ldg_addr")
        addr = self._resolve_descriptor_addr(Inst, ptr, addr, IRBuilder, IRRegs, _get_val)
        if Inst.opcodes[0] == "LD":
            # Generic (non-.E-global) loads may hit a materialized shared ptr
            addr = self._redirect_shared_window(IRBuilder, addr, "ldg")
            import os as _os
            if _os.environ.get("SLIFTER_OSTORE_LIFT"):
                i64 = self.ir.IntType(64)
                _a = addr
                if isinstance(_a.type, self.ir.PointerType):
                    _a = IRBuilder.ptrtoint(_a, i64, "oload_dbg_a")
                elif getattr(_a.type, "width", 64) != 64:
                    _a = (IRBuilder.zext if _a.type.width < 64 else IRBuilder.trunc)(
                        _a, i64, "oload_dbg_a")
                _w = 100 + (len(dest.reg_group) if dest.is_reg_group else 1)
                self.run_intrinsic("slifter_ostore_dbg", IRBuilder,
                                   [_a, self.ir.Constant(self.ir.IntType(32), _w)],
                                   "oload_dbg")

        # Check for sub-32-bit load width from opcode suffixes
        # LDG.E.U8, LDG.E.S8, LDG.E.U16, LDG.E.S16
        load_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        load_bits = None
        is_signed = False
        for opc in Inst.opcodes:
            if opc in load_width_map:
                load_bits = load_width_map[opc]
                is_signed = opc.startswith("S")
                break

        if dest.is_reg_group:
            # Vector load: use <N x i32> instead of iN for better x86 codegen
            reg_count = len(dest.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            ptr_name = f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr"
            addr_ptr = _as_pointer(addr, vec_ty, ptr_name)
            vec_val = IRBuilder.load(addr_ptr, "ldg_vec")
            self._vec_set_val(IRBuilder, IRRegs, dest, vec_val)
        elif load_bits:
            # Sub-32-bit load: load narrow type, then extend to i32
            narrow_ty = self.ir.IntType(load_bits)
            addr_ptr = _as_pointer(
                addr,
                narrow_ty,
                f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr"
            )
            val = IRBuilder.load(addr_ptr, "ldg", typ=narrow_ty)
            i32 = self.ir.IntType(32)
            if is_signed:
                val = IRBuilder.sext(val, i32, "ldg_sext")
            else:
                val = IRBuilder.zext(val, i32, "ldg_zext")
            self._set_val(IRBuilder, IRRegs, dest, val)
        else:
            pointee_ty = dest.get_ir_type(self)
            addr_ptr = _as_pointer(
                addr,
                pointee_ty,
                f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr"
            )
            val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
            self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("STG", "ST")
    def lift_stg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        # SM90 descriptor addressing: STG.E desc[URx][Ry.64+offset], Rsrc
        addr = _get_val(ptr, "stg_addr")
        addr = self._resolve_descriptor_addr(Inst, ptr, addr, IRBuilder, IRRegs, _get_val)
        if Inst.opcodes[0] == "ST":
            # Generic (non-.E-global) stores may hit a materialized shared ptr
            addr = self._redirect_shared_window(IRBuilder, addr, "stg")
            import os as _os
            if _os.environ.get("SLIFTER_OSTORE_LIFT"):
                i64 = self.ir.IntType(64)
                _a = addr
                if isinstance(_a.type, self.ir.PointerType):
                    _a = IRBuilder.ptrtoint(_a, i64, "ostore_dbg_a")
                elif getattr(_a.type, "width", 64) != 64:
                    _a = (IRBuilder.zext if _a.type.width < 64 else IRBuilder.trunc)(
                        _a, i64, "ostore_dbg_a")
                _w = len(val.reg_group) if val.is_reg_group else 1
                self.run_intrinsic("slifter_ostore_dbg", IRBuilder,
                                   [_a, self.ir.Constant(self.ir.IntType(32), _w)],
                                   "ostore_dbg")

        # Lockstep turnstile (CULIFTER_LOCKSTEP=1): a GLOBAL store executed by
        # several lanes of one warp that alias the same global cell with NO block
        # barrier between the writers is ordered on HW only by warp SIMT lockstep
        # (program/PC order within the converged warp). The wrapper's free-running
        # lane threads desync and reorder them -> last-writer race (the generic
        # block-barrier/global-store analog of the warp-reduction member-set bug;
        # localized in cuDNN compute_alphas_and_betas<float>'s racing alphas[1],
        # where init alpha[s1][t0] and recursion alpha[s0][t1] alias). Bracketing
        # the store retires it in warp round-order, the same contract the shared
        # LDS/STS accesses and shared atomics already use (see _turn_begin). Opt-in
        # -> no default-path change. Generic ST redirected into the shared window
        # is left to the shared turnstile path which already owns that ordering.
        _stg_turn = (Inst.opcodes[0] == "STG")
        if _stg_turn:
            self._turn_begin(IRBuilder)

        # Check for sub-32-bit store width from opcode suffixes
        # STG.E.U8, STG.E.U16, STG.E.S8, STG.E.S16
        store_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        store_bits = None
        for opc in Inst.opcodes:
            if opc in store_width_map:
                store_bits = store_width_map[opc]
                break

        if val.is_reg_group and not store_bits:
            # Vector store: use <N x i32> instead of iN for better x86 codegen
            vec_val = self._vec_get_val(IRBuilder, IRRegs, val, "stg_val")
            reg_count = len(val.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            ptr_name = f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "stg_addr_ptr"
            addr_ptr = _as_pointer(addr, vec_ty, ptr_name)
            IRBuilder.store(vec_val, addr_ptr)
        else:
            v = _get_val(val, "stg_val")

            if val.is_reg_group:
                pointee_ty = self.ir.IntType(len(val.reg_group) * 32)
            elif store_bits:
                pointee_ty = self.ir.IntType(store_bits)
            else:
                pointee_ty = v.type

            # Truncate value if storing sub-32-bit
            if store_bits:
                i32 = self.ir.IntType(32)
                # If value is float/half, bitcast to integer first
                if isinstance(v.type, (self.ir.FloatType, self.ir.HalfType)):
                    if isinstance(v.type, self.ir.HalfType):
                        v = IRBuilder.bitcast(v, self.ir.IntType(16), "stg_h2i")
                        v = IRBuilder.zext(v, i32, "stg_h2i32")
                    else:
                        v = IRBuilder.bitcast(v, i32, "stg_f2i")
                if hasattr(v.type, 'width') and v.type.width > store_bits:
                    v = IRBuilder.trunc(v, self.ir.IntType(store_bits), "stg_trunc")

            addr_ptr = _as_pointer(
                addr,
                pointee_ty,
                f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "stg_addr_ptr"
                )
            # Bitcast value if types don't match (e.g., double stored via i64 register pair)
            if v.type != pointee_ty:
                v = IRBuilder.bitcast(v, pointee_ty, "stg_val_cast")
            IRBuilder.store(v, addr_ptr)

        if _stg_turn:
            self._turn_end(IRBuilder)


    def _turn_begin(self, IRBuilder):
        if _LOCKSTEP:
            self.run_intrinsic("warp_turn_begin", IRBuilder, [])

    def _turn_end(self, IRBuilder):
        if _LOCKSTEP:
            self.run_intrinsic("warp_turn_end", IRBuilder, [])

    def _coerce_gep_index(self, val, IRBuilder, name):
        """A shared-memory address is an integer offset by construction. If type
        inference labeled the value as floating point (e.g. when seeds are
        removed in the type-degradation study), reinterpret its bits as an
        integer so it is a valid GEP index — LLVM rejects a float GEP index."""
        if isinstance(val.type, self.ir.IntType):
            return val
        fbits = {self.ir.HalfType: 16, self.ir.FloatType: 32,
                 self.ir.DoubleType: 64}.get(type(val.type))
        if fbits is not None:
            return IRBuilder.bitcast(val, self.ir.IntType(fbits), name)
        return val

    @lift_for("LDS")
    def lift_lds(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        self._turn_begin(IRBuilder)
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "lds_addr")
        addr = self._coerce_gep_index(addr, IRBuilder, "lds_addr_int")
        # DEBUG (gated, lift-time): dump the EFFECTIVE shared-memory offset of any
        # LDS that reads the FA3 bwd LSE/dPsum smem region (immediate ~0x2e000), to
        # localize where the consumer actually reads vs where UBLKCP wrote the tile.
        import os as _lds_dbg_os
        if _lds_dbg_os.environ.get("SLIFTER_LDS_DUMP") or _lds_dbg_os.environ.get("SLIFTER_LSE_TRACE"):
            try:
                _off_imm = int(getattr(ptr, "offset", 0) or 0)
            except (ValueError, TypeError):
                _off_imm = 0
            if 0x2d000 <= _off_imm <= 0x30000:
                self.run_intrinsic("slifter_lds_dump", IRBuilder,
                                   [addr, self.ir.Constant(self.ir.IntType(32), _off_imm)],
                                   "lds_dump")
        addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "lds_shared_addr")

        # Check for sub-32-bit load width from opcode suffixes (LDS.U8, LDS.S16, etc.)
        load_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        load_bits = None
        is_signed = False
        for opc in Inst.opcodes:
            if opc in load_width_map:
                load_bits = load_width_map[opc]
                is_signed = opc.startswith("S")
                break

        if load_bits:
            narrow_ty = self.ir.IntType(load_bits)
            if addr.type.pointee != narrow_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(narrow_ty),
                    "lds_shared_addr_cast",
                )
            val = IRBuilder.load(addr, "lds", typ=narrow_ty)
            i32 = self.ir.IntType(32)
            if is_signed:
                val = IRBuilder.sext(val, i32, "lds_sext")
            else:
                val = IRBuilder.zext(val, i32, "lds_zext")
            self._set_val(IRBuilder, IRRegs, dest, val)
        elif dest.is_reg_group:
            # Vector load for .128/.64 shared memory access
            reg_count = len(dest.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            if addr.type.pointee != vec_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(vec_ty),
                    "lds_shared_addr_cast",
                )
            vec_val = IRBuilder.load(addr, "lds_vec")
            self._vec_set_val(IRBuilder, IRRegs, dest, vec_val)
        else:
            if addr.type.pointee != dest.get_ir_type(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(dest.get_ir_type(self)),
                    "lds_shared_addr_cast",
                )
            val = IRBuilder.load(addr, "lds", typ=dest.get_ir_type(self))
            self._set_val(IRBuilder, IRRegs, dest, val)
        self._turn_end(IRBuilder)


    @lift_for("STS")
    def lift_sts(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        self._turn_begin(IRBuilder)
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "sts_addr")
        addr = self._coerce_gep_index(addr, IRBuilder, "sts_addr_int")
        # DEBUG (gated SLIFTER_LSE_TRACE): record any STS effective offset so a stray
        # scalar store clobbering the LSE/dPsum window is caught in the unified seq
        # stream (runtime filters by the 0x2e000.. region). Lift-time gated -> prod
        # byte-identical.
        import os as _sts_dbg_os
        if _sts_dbg_os.environ.get("SLIFTER_LSE_TRACE"):
            self.run_intrinsic("slifter_sts_dump", IRBuilder,
                               [addr,
                                self.ir.Constant(self.ir.PointerType(self.ir.IntType(8)), None),
                                self.ir.Constant(self.ir.IntType(32), 4)],
                               "sts_dump")
        # DEBUG (gated SLIFTER_DQACC_STS): record the per-thread dQ_accum STS write
        # (raw smem offset + 4 stored f32 values) so the (smem_flat -> query,hdim)
        # decode map for the dQ accumulator can be recovered by value-matching against
        # the CPU reference. Lift-time gated -> production byte-identical.
        _dqacc_dump = _sts_dbg_os.environ.get("SLIFTER_DQACC_STS") and val.is_reg_group
        _dqacc_raw_off = addr if _dqacc_dump else None
        addr = IRBuilder.gep(self.SharedMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "sts_shared_addr")

        # Check for sub-32-bit store width from opcode suffixes (STS.U8, STS.S16, etc.)
        store_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        store_bits = None
        for opc in Inst.opcodes:
            if opc in store_width_map:
                store_bits = store_width_map[opc]
                break

        if val.is_reg_group and not store_bits:
            # Vector store for .128/.64 shared memory access
            vec_val = self._vec_get_val(IRBuilder, IRRegs, val, "sts_val")
            reg_count = len(val.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            if addr.type.pointee != vec_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(vec_ty),
                    "sts_shared_addr_cast",
                )
            if _dqacc_raw_off is not None:
                # Spill the vector to a stack slot and pass an i8* + raw offset.
                slot = IRBuilder.alloca(vec_ty, name="dqacc_dump_slot")
                IRBuilder.store(vec_val, slot)
                slot_i8 = IRBuilder.bitcast(
                    slot, self.ir.PointerType(self.ir.IntType(8)), "dqacc_dump_i8")
                self.run_intrinsic("slifter_dqacc_sts_dump", IRBuilder,
                                   [_dqacc_raw_off, slot_i8,
                                    self.ir.Constant(self.ir.IntType(32), reg_count)],
                                   "dqacc_sts_dump")
            IRBuilder.store(vec_val, addr)
        else:
            v = _get_val(val, "sts_val")

            if store_bits:
                narrow_ty = self.ir.IntType(store_bits)
                # Float inputs must go through bitcast-to-int before truncation;
                # IRBuilder.trunc is int-only, and without this the fallback path
                # hits "cannot store float to i16*" for e.g. STS.16 with an F32 src.
                if isinstance(v.type, (self.ir.FloatType, self.ir.DoubleType, self.ir.HalfType)):
                    src_bits = {self.ir.FloatType: 32, self.ir.DoubleType: 64,
                                self.ir.HalfType: 16}[type(v.type)]
                    v = IRBuilder.bitcast(v, self.ir.IntType(src_bits), "sts_val_as_int")
                if isinstance(v.type, self.ir.IntType) and v.type.width > store_bits:
                    v = IRBuilder.trunc(v, narrow_ty, "sts_trunc")
                if addr.type.pointee != narrow_ty:
                    addr = IRBuilder.bitcast(
                        addr,
                        self.ir.PointerType(narrow_ty),
                        "sts_shared_addr_cast",
                    )
            else:
                if addr.type.pointee != val.get_ir_type(self):
                    addr = IRBuilder.bitcast(
                        addr,
                        self.ir.PointerType(val.get_ir_type(self)),
                        "sts_shared_addr_cast",
                    )
            IRBuilder.store(v, addr)
        self._turn_end(IRBuilder)


    @lift_for("MOVM")
    def lift_movm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """MOVM.16.MT88 — warp-collective 8x8 b16 register transpose
        (PTX movmatrix.sync.m8n8.trans.b16). The wrapper runs warps
        concurrently, so the transpose must really exchange across lanes:
        route to the movm_mt88_warp runtime collective. Non-transposing
        forms remain pass-through.
        """
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "movm")
        if "MT88" in Inst.opcodes:
            i32 = self.ir.IntType(32)
            val = self._insert_type_cast_if_needed(IRBuilder, val, i32, "movm_i32")
            ext = self.ir_functions.get("movm_mt88_warp")
            if ext is None:
                ext = self.ir.Function(
                    self.llvm_module,
                    self.ir.FunctionType(i32, [i32], False),
                    "movm_mt88_warp")
                self.ir_functions["movm_mt88_warp"] = ext
            val = IRBuilder.call(ext, [val], "movm_t")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("LDSM")
    def lift_ldsm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """LDSM.16.M88[.N] / LDSM.16.MT88[.N] Rdst, [Raddr] — PTX ldmatrix.

        ldmatrix is warp-collective: each lane addresses one 16-byte matrix
        ROW (lanes 8m..8m+7 address the rows of matrix m), but each lane's
        result regs are fragments gathered from rows addressed by OTHER
        lanes (reg m of lane L = 4-byte slice (L&3) of the row from lane
        8m+(L>>2); MT88 is the transposed gather). A per-lane contiguous
        load is a warp-wide fragment permutation (vs the ldsm_x4 GPU
        capture only 8/512 elements coincide) which feeds tensor-core MMAs
        permuted A/B. So: load this lane's full 16-byte row (HW reads whole
        rows regardless of N) and route through the ldsm_m88_warp runtime
        collective, which redistributes fragments for all 32 lanes.
        """
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldsm_addr")

        # Determine count from opcode: LDSM.16.M88.N or LDSM.16.MT88.N
        count = 1
        last_opc = Inst.opcodes[-1]
        if last_opc.isdigit():
            count = int(last_opc)
        trans = "MT88" in Inst.opcodes

        i32 = self.ir.IntType(32)

        # Defensive: GEP index must be integer
        if isinstance(addr.type, self.ir.FloatType):
            import logging
            logging.warning(f"TRIAGE: float addr in LDSM at PC={Inst.offset:#x}")
            addr = IRBuilder.bitcast(addr, i32, "ldsm_addr_fix")

        # Load the lane's 16-byte row (4 x i32) at its own address.
        row_words = []
        for k in range(4):
            woff = addr if k == 0 else IRBuilder.add(
                addr, self.ir.Constant(addr.type, 4 * k), f"ldsm_off{k}")
            shared_addr = IRBuilder.gep(
                self.SharedMem,
                [self.ir.Constant(i32, 0), woff],
                f"ldsm_row_addr{k}",
            )
            if shared_addr.type.pointee != i32:
                shared_addr = IRBuilder.bitcast(
                    shared_addr, self.ir.PointerType(i32), f"ldsm_row_cast{k}"
                )
            row_words.append(IRBuilder.load(shared_addr, f"ldsm_row{k}", typ=i32))

        intrinsic = f"ldsm_{'mt88' if trans else 'm88'}_x{count}"
        val = self.run_intrinsic(intrinsic, IRBuilder, row_words, "ldsm")
        self._set_val(IRBuilder, IRRegs, dest, val)


    @staticmethod
    def _stsm_src_from_deinterleave(Inst, val_op):
        """Return True if the STSM source operand is (transitively) produced by
        the e4m3 byte de-interleave PRMT(0x6420 / 0x7531) — i.e. this is the FA3
        fp8 in-kernel V-transpose store, not the fp32->bf16 O-epilogue repack.

        Walks reaching_defs from each source sub-register (through MOV chains)
        looking for a PRMT/UPRMT whose selector immediate is 0x6420 or 0x7531.
        Robust + general: the O-epilogue STSM's sources come from F2FP/SEL and
        never reach that PRMT, so they correctly stay on the b16 path. Any
        malformed chain is treated as 'not a V-transpose' (returns False) so the
        discriminator can never break STSM lifting."""
        try:
            if not getattr(Inst, "reaching_defs", None):
                return False
            srcs = list(val_op.reg_group) if getattr(val_op, "is_reg_group", False) else [val_op]
            seen = set()
            # BFS over the reaching-def chain, bounded depth (MOV relays only).
            frontier = [(Inst, s, 0) for s in srcs]
            while frontier:
                cur_inst, op, depth = frontier.pop()
                if depth > 6:
                    continue
                rd = getattr(cur_inst, "reaching_defs", None)
                def_inst = rd.get(op) if rd else None
                if def_inst is None or def_inst is cur_inst:
                    continue
                key = id(def_inst)
                if key in seen:
                    continue
                seen.add(key)
                opc0 = def_inst.opcodes[0] if def_inst.opcodes else ""
                if opc0 in ("PRMT", "UPRMT"):
                    for u in def_inst.get_uses():
                        if getattr(u, "is_immediate", False) and \
                           getattr(u, "immediate_value", None) in (0x6420, 0x7531):
                            return True
                    # a non-de-interleave PRMT is a dead end (don't recurse into it).
                    continue
                # Relay only through pure value-moving ops to the real producer.
                # (NOT F2FP/SEL: those mark the O-epilogue fp32->bf16 repack, whose
                # source must NOT be misread as the V-transpose.)
                if opc0 in ("MOV", "UMOV"):
                    for u in def_inst.get_uses():
                        if getattr(u, "is_reg", False):
                            frontier.append((def_inst, u, depth + 1))
        except Exception:
            return False
        return False

    @lift_for("STSM")
    def lift_stsm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """STSM.16.[M|MT]88[.N] [Raddr], Rsrc — PTX stmatrix (warp-collective).

        stmatrix is the warp-collective inverse of ldmatrix: each lane
        addresses one 16-byte matrix ROW (lanes 8m..8m+7 the rows of matrix
        m), but the row written there gathers fragments from ACROSS the warp
        (word w of the row at lane (8m+r)'s address = src reg m of lane
        4r+w; MT88 stores the transposed gather). A per-lane contiguous
        store is a warp-wide fragment permutation — the symmetric STSM-side
        twin of the LDSM bug — wrong whenever fragments round-trip through
        shared memory. Lanes >= 8*count have NO target row ("addresses of
        the remaining threads are ignored", PTX ISA), so unlike LDSM the
        lifted IR must not touch memory itself: it parks the lane's own
        target-row pointer + N source regs in the stsm_m88_warp runtime
        collective (kernel_wrapper_75.h), whose last arriver performs all
        stores through the parked pointers of lanes 0..8*count-1 only.
        """
        uses = Inst.get_uses()
        ptr, val_op = uses[0], uses[1]
        addr = _get_val(ptr, "stsm_addr")

        # Determine count from opcode: STSM.16.M88.N or STSM.16.MT88.N
        count = 1
        last_opc = Inst.opcodes[-1]
        if last_opc.isdigit():
            count = int(last_opc)
        trans = "MT88" in Inst.opcodes
        # FP8 (e4m3/e5m2) in-kernel V-transpose: the store is STSM.16.M88.4
        # (non-trans), but the preceding LDSM.16.MT88.4 + PRMT(0x6420/0x7531)
        # de-interleave make the whole chain an 8-bit transpose. Treat each
        # gathered fragment at BYTE granularity (trans==2) so the net Vt is a
        # correct e4m3 transpose. fp16/bf16 kernels never set _fa_fp8, so the
        # b16 transpose path (trans 0/1) is byte-identical for them.
        #
        # CRITICAL: an fp8 FA kernel has TWO kinds of STSM.16.M88.4 (non-trans):
        #   1. the V-transpose store, whose 4 source regs are produced by the
        #      e4m3 byte de-interleave PRMT(0x6420/0x7531) — needs trans_mode=2;
        #   2. the O EPILOGUE store, which repacks the fp32->bf16 O fragment
        #      (sources come from F2FP/SEL, NOT the de-interleave PRMT) — needs
        #      the ordinary b16 path (trans_mode=0). Gating trans_mode=2 on the
        #      whole-function _fa_fp8 flag wrongly byte-scrambled the O store
        #      (the V=ones "hdim blocks 1,3 zero" signature was the O epilogue,
        #      not the V read). So gate per-instruction: only the STSM whose
        #      source reaches a de-interleave PRMT is the V-transpose.
        is_fp8_vt = (getattr(self, "_fa_fp8", False) and not trans and count == 4
                     and self._stsm_src_from_deinterleave(Inst, val_op))
        trans_mode = (2 if is_fp8_vt else (1 if trans else 0))

        i32 = self.ir.IntType(32)

        # Defensive: GEP index must be integer
        if isinstance(addr.type, self.ir.FloatType):
            import logging
            logging.warning(f"TRIAGE: float addr in STSM at PC={Inst.offset:#x}")
            addr = IRBuilder.bitcast(addr, i32, "stsm_addr_fix")

        # DEBUG (gated SLIFTER_LSE_TRACE): record any STSM effective offset (per-lane
        # base) so a matrix store landing in the LSE/dPsum window is caught in the
        # unified seq stream (runtime filters by region). Lift-time gated.
        import os as _stsm_dbg_os
        if _stsm_dbg_os.environ.get("SLIFTER_LSE_TRACE"):
            self.run_intrinsic("slifter_sts_dump", IRBuilder,
                               [addr,
                                self.ir.Constant(self.ir.PointerType(self.ir.IntType(8)), None),
                                self.ir.Constant(i32, 16)],
                               "stsm_dump")

        # The lane's own 16-byte target row in shared memory.
        shared_addr = IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(i32, 0), addr],
            "stsm_shared_addr",
        )
        if shared_addr.type.pointee != i32:
            shared_addr = IRBuilder.bitcast(
                shared_addr, self.ir.PointerType(i32), "stsm_shared_cast"
            )

        # Split the packed source value into count i32 words (pad with 0).
        v = _get_val(val_op, "stsm_val")
        total_ty = self.ir.IntType(count * 32)
        if v.type != total_ty:
            v = self._insert_type_cast_if_needed(IRBuilder, v, total_ty, "stsm_val_cast")
        words = []
        for k in range(count):
            w = v
            if count > 1:
                if k:
                    w = IRBuilder.lshr(w, self.ir.Constant(total_ty, 32 * k),
                                       f"stsm_w{k}_shift")
                w = IRBuilder.trunc(w, i32, f"stsm_w{k}")
            words.append(w)
        while len(words) < 4:
            words.append(self.ir.Constant(i32, 0))

        ext = self.ir_functions.get("stsm_m88_warp")
        if ext is None:
            ext_ty = self.ir.FunctionType(
                self.ir.VoidType(),
                [self.ir.PointerType(i32)] + [i32] * 6, False)
            ext = self.ir.Function(self.llvm_module, ext_ty, "stsm_m88_warp")
            self.ir_functions["stsm_m88_warp"] = ext
        IRBuilder.call(ext, [shared_addr] + words
                       + [self.ir.Constant(i32, count),
                          self.ir.Constant(i32, trans_mode)])


    @lift_for("LDGSTS")
    def lift_ldgsts(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """LDGSTS.E[.64|.128] [shared_offset], desc[UR][global_addr.64], P
        Async copy from global to shared. On x86: synchronous load+store.
        Operand layout (all uses, no defs):
          uses[0] = shared memory offset  ([R7])
          uses[1] = global memory address  (desc[UR6][R2.64])
          uses[2] = guard predicate        (P6 / !P0 / PT)  -- optional

        The trailing predicate is a *first-class operand* of LDGSTS: the
        async copy executes only when the predicate is true. HW skips it
        otherwise — and CUTLASS (36_gather_scatter_fusion) relies on that:
        the global address is derived from a per-tile gather index that is
        out-of-range (or never initialized -> NULL after 32+32 packing) for
        threads whose predicate is false. Dropping the predicate made the
        load unconditional and dereferenced that NULL/OOB address -> SIGSEGV.

        We keep the lowering branch-free (no new basic blocks, which would
        desync the per-CFG-block lifting loop): when the predicate is false,
        load from a known-safe pointer (SharedMem base — always mapped) and
        leave shared memory unchanged. Both halves use plain `select`, the
        same pattern lift_sdiv uses to make a faulting op safe.
        """
        uses = Inst.get_uses()
        shared_off_op = uses[0]
        global_addr_op = uses[1]
        # The guard predicate (if any) is the trailing predicate-register /
        # PT operand. Locate it by kind rather than a fixed index so encodings
        # without an explicit predicate simply yield None (unconditional).
        pred_op = None
        for op in uses[2:]:
            if getattr(op, "is_predicate_reg", False) or getattr(op, "is_pt", False):
                pred_op = op
                break

        # Determine transfer width from opcode modifiers
        # LDGSTS.E = 32-bit (default), .64 = 64-bit, .128 = 128-bit
        xfer_elems = 1
        if "128" in Inst.opcodes:
            xfer_elems = 4
        elif "64" in Inst.opcodes:
            xfer_elems = 2

        i32 = self.ir.IntType(32)
        if xfer_elems > 1:
            xfer_ty = self.ir.VectorType(i32, xfer_elems)
        else:
            xfer_ty = i32

        # Resolve the guard predicate to an i1. `PT` / no-predicate -> always
        # copy (skip the guard entirely). `_get_val` already folds the `!`
        # modifier for !P-style operands.
        guard = None
        if pred_op is not None and not (
            getattr(pred_op, "is_pt", False) and not getattr(pred_op, "is_not_reg", False)
        ):
            guard = _get_val(pred_op, "ldgsts_pred")
            guard = self._normalize_i1_in_curr_bb(IRBuilder, guard, "ldgsts_guard")

        # Shared-memory destination pointer (always valid; used both as the
        # store target and as the safe load source when the predicate is off).
        shared_off = _get_val(shared_off_op, "ldgsts_soff")
        # Defensive: GEP index must be integer. If type analysis failed to
        # constrain the register, bitcast float -> i32 rather than crashing.
        if isinstance(shared_off.type, self.ir.FloatType):
            import logging
            logging.warning(f"TRIAGE: float offset in LDGSTS at PC={Inst.offset:#x}")
            shared_off = IRBuilder.bitcast(shared_off, i32, "ldgsts_soff_fix")
        shared_addr = IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(i32, 0), shared_off],
            "ldgsts_shared_addr",
        )
        shared_ptr = IRBuilder.bitcast(
            shared_addr, self.ir.PointerType(xfer_ty), "ldgsts_sptr"
        )

        # Step 1: Load from global memory. When guarded, redirect the load to
        # the (always-mapped) shared destination if the predicate is false so
        # the dereference can never fault on a stale/NULL gather address.
        global_addr = _get_val(global_addr_op, "ldgsts_gaddr")
        import os as _os
        if _os.environ.get("SLIFTER_LDGSTS_LIFT"):
            i64 = self.ir.IntType(64)
            _ga = self._insert_type_cast_if_needed(IRBuilder, global_addr, i64, "ldgsts_dbg_ga")
            _so = self._insert_type_cast_if_needed(IRBuilder, shared_off, i32, "ldgsts_dbg_so")
            self.run_intrinsic("slifter_ldgsts_dbg", IRBuilder, [_ga, _so], "ldgsts_dbg")
        global_ptr = _as_pointer(global_addr, xfer_ty, "ldgsts_gptr")
        if guard is not None:
            global_ptr = IRBuilder.select(
                guard, global_ptr, shared_ptr, "ldgsts_safe_gptr"
            )
        global_val = IRBuilder.load(global_ptr, "ldgsts_gload")

        # Step 2: Store to shared memory. When the predicate is false the HW
        # copy is a no-op, so write back the existing shared value instead of
        # the (guard-off, meaningless) loaded one.
        if guard is not None:
            old_val = IRBuilder.load(shared_ptr, "ldgsts_sold")
            global_val = IRBuilder.select(
                guard, global_val, old_val, "ldgsts_store_val"
            )
        IRBuilder.store(global_val, shared_ptr)


    @lift_for("LDL")
    def lift_ldl(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldl_addr")
        # SASS LDL takes a byte-addressed local-memory offset (R1 is the
        # byte stack pointer + literal byte offset). LocalMem is [N x i8],
        # so the GEP index is the byte address directly — no scaling.
        # (The NVVM lifter has always done this; the CPU lifter had a stray
        # "mul addr, 8" from an early experiment that silently 8x-stretched
        # every local frame. Kernels with small frames kept working because
        # both LDL and STL stretched consistently and stayed inside the 32 KB
        # local_mem; larger frames overflowed it and corrupted the heap.)
        addr = self._coerce_gep_index(addr, IRBuilder, "ldl_addr_int")
        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "ldl_local_addr")

        # Check for sub-32-bit load width from opcode suffixes (LDL.U8, LDL.S16, etc.)
        load_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        load_bits = None
        is_signed = False
        for opc in Inst.opcodes:
            if opc in load_width_map:
                load_bits = load_width_map[opc]
                is_signed = opc.startswith("S")
                break

        if load_bits:
            narrow_ty = self.ir.IntType(load_bits)
            if addr.type.pointee != narrow_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(narrow_ty),
                    "ldl_local_addr_cast",
                )
            val = IRBuilder.load(addr, "ldl", typ=narrow_ty)
            i32 = self.ir.IntType(32)
            if is_signed:
                val = IRBuilder.sext(val, i32, "ldl_sext")
            else:
                val = IRBuilder.zext(val, i32, "ldl_zext")
            self._set_val(IRBuilder, IRRegs, dest, val)
        elif dest.is_reg_group:
            # Vector load for .128/.64 local memory access
            reg_count = len(dest.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            if addr.type.pointee != vec_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(vec_ty),
                    "ldl_local_addr_cast",
                )
            vec_val = IRBuilder.load(addr, "ldl_vec")
            self._vec_set_val(IRBuilder, IRRegs, dest, vec_val)
        else:
            if addr.type.pointee != dest.get_ir_type(self):
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(dest.get_ir_type(self)),
                    "ldl_local_addr_cast",
                )
            val = IRBuilder.load(addr, "ldl", typ=dest.get_ir_type(self))
            self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("STL")
    def lift_stl(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "stl_addr")
        # See lift_ldl: byte-addressed local memory, no scaling.
        addr = self._coerce_gep_index(addr, IRBuilder, "stl_addr_int")
        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "stl_local_addr")

        # Check for sub-32-bit store width from opcode suffixes (STL.U8, STL.S16, etc.)
        store_width_map = {"U8": 8, "S8": 8, "U16": 16, "S16": 16}
        store_bits = None
        for opc in Inst.opcodes:
            if opc in store_width_map:
                store_bits = store_width_map[opc]
                break

        if val.is_reg_group and not store_bits:
            # Vector store for .128/.64 local memory access
            vec_val = self._vec_get_val(IRBuilder, IRRegs, val, "stl_val")
            reg_count = len(val.reg_group)
            vec_ty = self.ir.VectorType(self.ir.IntType(32), reg_count)
            if addr.type.pointee != vec_ty:
                addr = IRBuilder.bitcast(
                    addr,
                    self.ir.PointerType(vec_ty),
                    "stl_local_addr_cast",
                )
            IRBuilder.store(vec_val, addr)
        else:
            v = _get_val(val, "stl_val")

            if store_bits:
                narrow_ty = self.ir.IntType(store_bits)
                if hasattr(v.type, 'width') and v.type.width > store_bits:
                    v = IRBuilder.trunc(v, narrow_ty, "stl_trunc")
                if addr.type.pointee != narrow_ty:
                    addr = IRBuilder.bitcast(
                        addr,
                        self.ir.PointerType(narrow_ty),
                        "stl_local_addr_cast",
                    )
            else:
                if addr.type.pointee != val.get_ir_type(self):
                    addr = IRBuilder.bitcast(
                        addr,
                        self.ir.PointerType(val.get_ir_type(self)),
                        "stl_local_addr_cast",
                    )
            IRBuilder.store(v, addr)


    @lift_for("CALLEXT")
    def lift_callext(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """Extern device-runtime call resolved from the bank-4 pointer table
        (ResolveIndirectCall). Uses: R5:R4 / R7:R6 = first two 64-bit params
        (CUDA device call ABI); def R4 = i32 return.

        __assertfail aborts the process (assert semantics); other callees
        (vprintf) are declared `i32 @<name>(i8*, i8*)` and provided by the
        host runtime (printf_runtime.c / kernel_wrapper). NOTE: the param
        translation passes the raw 64-bit register values as pointers —
        valid for c[0x4]-sourced host pointers, but a local_mem-relative
        arg buffer (vprintf's valist) is not translated yet; the host
        vprintf guards against that by validating fmt first.
        """
        name = Inst.ext_callee if hasattr(Inst, "ext_callee") else Inst.opcodes[1]
        i8ptr = self.ir.PointerType(self.ir.IntType(8))
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        uses = Inst.get_uses()
        if name == "__assertfail":
            # Device assert: print-and-continue via the host __assertfail
            # (kernel-trap semantics, NOT process abort — see the wrapper).
            # Message/file pointers come from unmaterialized bank-4 $str
            # slots today, so pass NULLs; the wrapper prints "?" for those.
            fn = self.ir_functions.get("__assertfail")
            if fn is None:
                fn_ty = self.ir.FunctionType(
                    self.ir.VoidType(),
                    [i8ptr, i8ptr, i32, i8ptr, i64])
                fn = self.ir.Function(self.llvm_module, fn_ty, "__assertfail")
                self.ir_functions["__assertfail"] = fn
            nullp = self.ir.Constant(i8ptr, None)
            IRBuilder.call(fn, [nullp, nullp, self.ir.Constant(i32, 0),
                                nullp, self.ir.Constant(i64, 0)], "")
            return
        arg_vals = []
        for u in uses[:2]:
            v = _get_val(u, "callext_arg")
            v = self._insert_type_cast_if_needed(IRBuilder, v, i64, "callext_arg_i64")
            arg_vals.append(IRBuilder.inttoptr(v, i8ptr, "callext_arg_ptr"))
        fn = self.ir_functions.get(name)
        if fn is None:
            fn_ty = self.ir.FunctionType(i32, [i8ptr, i8ptr])
            fn = self.ir.Function(self.llvm_module, fn_ty, name)
            self.ir_functions[name] = fn
        ret = IRBuilder.call(fn, arg_vals, "callext")
        self._set_val(IRBuilder, IRRegs, Inst.get_defs()[0], ret)

    @lift_for("CS2R")
    def lift_cs2r(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        ResOp = Inst.get_defs()[0]
        ValOp = Inst.get_uses()[0]
        if ResOp.is_reg_group:
            # 64-bit CS2R (pair dest, widened in sir/instruction.py). SRZ is
            # the common idiom; 64-bit SR values widen from the 32-bit read.
            if ValOp.is_rz:
                val64 = self.ir.Constant(self.ir.IntType(64), 0)
            else:
                offset = SR_TO_OFFSET.get(ValOp.name)
                if offset is not None:
                    v32 = self._load_const_mem_u32(IRBuilder, offset, f"cs2r_{offset:x}")
                else:
                    print(f"CS2R(64): Unknown special register {ValOp}")
                    v32 = self.ir.Constant(self.ir.IntType(32), 0)
                val64 = IRBuilder.zext(v32, self.ir.IntType(64), "cs2r64_zext")
            self._set_val(IRBuilder, IRRegs, ResOp, val64)
            return
        if ResOp.is_reg:
            dest_name = ResOp.get_ir_name(self)
            if ValOp.is_rz:
                IRVal = self.ir.Constant(self.ir.IntType(32), 0)
            else:
                offset = SR_TO_OFFSET.get(ValOp.name)
                if offset is not None:
                    IRVal = self._load_const_mem_u32(IRBuilder, offset, f"cs2r_{offset:x}")
                elif ValOp.is_warp_size:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 32)
                elif ValOp.is_active_mask:
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
                elif ValOp.is_lane_mask_eq:
                    IRVal = self._emit_lane_mask(IRBuilder, "eq", "cs2r_lanemask_eq")
                elif ValOp.is_lane_mask_le:
                    IRVal = self._emit_lane_mask(IRBuilder, "le", "cs2r_lanemask_le")
                elif ValOp.is_lane_mask_lt:
                    IRVal = self._emit_lane_mask(IRBuilder, "lt", "cs2r_lanemask_lt")
                elif ValOp.is_lane_mask_ge:
                    IRVal = self._emit_lane_mask(IRBuilder, "ge", "cs2r_lanemask_ge")
                elif ValOp.is_lane_mask_gt:
                    IRVal = self._emit_lane_mask(IRBuilder, "gt", "cs2r_lanemask_gt")
                else:
                    print(f"CS2R: Unknown special register {ValOp}")
                    IRVal = self.ir.Constant(self.ir.IntType(32), 0)

            if dest_name in IRRegs and IRRegs[dest_name].type.is_pointer:
                IRBuilder.store(IRVal, IRRegs[dest_name])
            else:
                IRRegs[dest_name] = IRVal


    @lift_for("VOTE", "VOTEU")
    def lift_vote(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        # SM75 format: VOTE.ANY R7, PT, P0  (uses[0]=mask, uses[1]=predicate)
        # SM90 format: VOTE.ANY P0, P0      (uses[0]=predicate only, no mask)
        # VOTEU format: VOTEU.ANY UR5, UPT, PT  (uses[0]=UPT mask, uses[1]=PT pred)
        if len(uses) > 1:
            mask_op = uses[0]
            pred_val = _get_val(uses[1], "vote_pred")
            # When mask operand is PT/UPT (predicate-true), use full mask
            if mask_op.is_pt:
                mask_val = ir.Constant(ir.IntType(32), 0xFFFFFFFF)
            else:
                mask_val = _get_val(mask_op, "vote_mask")
        else:
            # SM90: single use is the predicate; mask is implicitly all-lanes
            mask_val = ir.Constant(ir.IntType(32), 0xFFFFFFFF)
            pred_val = _get_val(uses[0], "vote_pred")

        if pred_val.type != ir.IntType(1):
            zero = ir.Constant(pred_val.type, 0)
            pred_val = IRBuilder.icmp_unsigned("!=", pred_val, zero, "vote_pred_i1")

        if mask_val.type != ir.IntType(32):
            mask_val = self._insert_type_cast_if_needed(IRBuilder, mask_val, ir.IntType(32), "vote_mask")

        mode = Inst.opcodes[1].lower()

        # SM90 VOTE.ANY/ALL writes ballot mask to R-register dest and predicate
        # to P-register dest.  When the first def is an integer register (not a
        # predicate), emit vote_ballot (returns i32 bitmask) instead of
        # vote_any/all (returns i8 boolean — C++ `bool` ABI; the i8 zexts to
        # i32 cleanly when consumed as a predicate).
        site = self._collective_site(Inst)
        dest = defs[0]
        if not dest.is_predicate_reg and not dest.is_pt:
            # Integer register destination: emit ballot mask (i32)
            intrinsic_name = "vote_ballot"
            ballotVal = self.run_intrinsic(intrinsic_name, IRBuilder,
                                           [mask_val, pred_val, site], intrinsic_name)
            self._set_val(IRBuilder, IRRegs, dest, ballotVal)
            # If there's a second def (predicate), set it based on ballot != 0
            if len(defs) > 1 and defs[1].is_predicate_reg and not defs[1].is_pt:
                if mode == "any":
                    pred_res = IRBuilder.icmp_unsigned(
                        "!=", ballotVal,
                        ir.Constant(ir.IntType(32), 0), "vote_any_pred")
                elif mode == "all":
                    # all lanes true if ballot == full mask
                    pred_res = IRBuilder.icmp_unsigned(
                        "==", ballotVal,
                        ir.Constant(ir.IntType(32), 0xFFFFFFFF), "vote_all_pred")
                else:
                    pred_res = ir.Constant(ir.IntType(1), 1)
                self._set_val(IRBuilder, IRRegs, defs[1], pred_res)
        else:
            # Predicate-only destination (legacy): emit boolean vote
            intrinsic_name = f"vote_{mode}"
            voteVal = self.run_intrinsic(intrinsic_name, IRBuilder,
                                         [mask_val, pred_val, site], intrinsic_name)
            self._set_val(IRBuilder, IRRegs, dest, voteVal)


    @lift_for("REDUX")
    def lift_redux(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # REDUX.SUM UR8, R8 — warp-wide reduction into a uniform register.
        # The CTA threads run concurrently in the x86 wrapper, so this must
        # be a real cross-lane reduction (runtime warp_redux on the
        # history-matched group), NOT an identity: nvcc's atomic aggregation
        # pattern (REDUX.SUM ; VOTEU/UFLO leader election ; @leader REDG)
        # otherwise drops every non-leader lane's contribution (hecbench
        # mr-cuda val=0).
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "redux_src")

        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        op = next((m for m in mods if m in ("SUM", "MIN", "MAX", "AND", "OR", "XOR")), None)
        if op is None:
            raise UnsupportedInstructionException(f"Unsupported REDUX op: {Inst.opcodes}")
        if op in ("MIN", "MAX"):
            # Unsigned is the unprinted SASS default: __reduce_min_sync(u32)
            # disassembles to plain `REDUX.MIN`, the int overload to
            # `REDUX.MIN.S32` (verified nvcc 12.2 -arch=sm_90).
            name = f"redux_{op.lower()}_" + ("s32" if "S32" in mods else "u32")
        else:
            name = f"redux_{op.lower()}"
        res = self.run_intrinsic(name, IRBuilder, [val, self._collective_site(Inst)], name)
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("ATOM")
    def lift_atom(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        
        op_type = None
        data_type = None
        is_signed = False
        
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "EXCH", "INC", "DEC"]:
                op_type = mod
            elif mod.startswith("F16X2") or mod == "F16X2":
                data_type = "F16x2"
            elif mod.startswith("F32") or mod == "F":
                data_type = "F32"
            elif mod.startswith("F64"):
                data_type = "F64"
            elif mod.startswith("S32"):
                data_type = "S32"
                is_signed = True
            elif mod.startswith("U32"):
                data_type = "U32"
            elif mod.startswith("S64"):
                data_type = "S64"
                is_signed = True
            elif mod.startswith("U64"):
                data_type = "U64"
        
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown ATOM operation: {Inst.opcodes}")
        
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        
        dest_reg = None
        for d in defs:
            if d.is_writable_reg and not d.is_pt and not d.is_rz:
                dest_reg = d
                break
        
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"No memory operand found in ATOM: {Inst.opcodes}")
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "atom_addr")
        
        src_op = uses[-1]
        src_val = _get_val(src_op, "atom_src_val")
        
        if data_type == "F16x2":
            elem_ty = self.ir.IntType(32)
        elif data_type == "F32" or data_type is None:
            elem_ty = self.ir.FloatType()
        elif data_type == "F64":
            elem_ty = self.ir.DoubleType()
        elif data_type in ["S32", "U32"]:
            elem_ty = self.ir.IntType(32)
        elif data_type in ["S64", "U64"]:
            elem_ty = self.ir.IntType(64)
        else:
            elem_ty = self.ir.IntType(32)
        
        ptr = _as_pointer(addr_val, elem_ty, "atom_ptr")
        
        src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "atom_src")
        
        if op_type == "ADD":
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                result = IRBuilder.atomic_rmw("fadd", ptr, src_val, "seq_cst", "atom_fadd")
            else:
                result = IRBuilder.atomic_rmw("add", ptr, src_val, "seq_cst", "atom_add")
        elif op_type == "MIN":
            rmw_op = "min" if is_signed else "umin"
            result = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", "atom_min")
        elif op_type == "MAX":
            rmw_op = "max" if is_signed else "umax"
            result = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", "atom_max")
        elif op_type == "AND":
            result = IRBuilder.atomic_rmw("and", ptr, src_val, "seq_cst", "atom_and")
        elif op_type == "OR":
            result = IRBuilder.atomic_rmw("or", ptr, src_val, "seq_cst", "atom_or")
        elif op_type == "XOR":
            result = IRBuilder.atomic_rmw("xor", ptr, src_val, "seq_cst", "atom_xor")
        elif op_type == "EXCH":
            result = IRBuilder.atomic_rmw("xchg", ptr, src_val, "seq_cst", "atom_exch")
        elif op_type == "INC":
            one = self.ir.Constant(elem_ty, 1)
            result = IRBuilder.atomic_rmw("add", ptr, one, "seq_cst", "atom_inc")
        elif op_type == "DEC":
            one = self.ir.Constant(elem_ty, 1)
            result = IRBuilder.atomic_rmw("sub", ptr, one, "seq_cst", "atom_dec")
        else:
            raise UnsupportedInstructionException(f"Unsupported ATOM operation: {op_type}")
        
        if dest_reg:
            dest_ty = dest_reg.get_ir_type(self)
            result = self._insert_type_cast_if_needed(IRBuilder, result, dest_ty, "atom_result")
            self._set_val(IRBuilder, IRRegs, dest_reg, result)
        
        pred = defs[0] if defs else None
        if pred and pred.is_predicate_reg:
            self._set_val(IRBuilder, IRRegs, pred, self.ir.Constant(self.ir.IntType(1), 1))


    @lift_for("ATOMG")
    def lift_atomg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        # SM90: ATOMG.E.ADD desc[URx][Ry.64], Rsrc
        # For x86, descriptor register is ignored — Ry.64 is the address.
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        
        is_64bit = "64" in modifiers
        op_type = None
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "EXCH", "CAS", "CAST", "INC", "DEC"]:
                op_type = mod
                break
        
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown ATOMG operation: {Inst.opcodes}")
        
        dest_reg = None
        for d in defs:
            if d.is_writable_reg and not d.is_pt:
                dest_reg = d
                break
        
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"No memory operand found in ATOMG: {Inst.opcodes}")
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "atomg_addr")
        
        elem_ty = self.ir.IntType(64) if is_64bit else self.ir.IntType(32)
        ptr = _as_pointer(addr_val, elem_ty, "atomg_ptr")
        
        if op_type in ("CAS", "CAST"):
            cmp_val = _get_val(uses[1], "atomg_cmp")
            new_val = _get_val(uses[2], "atomg_new")

            cmp_val = self._insert_type_cast_if_needed(IRBuilder, cmp_val, elem_ty, "atomg_cmp")
            new_val = self._insert_type_cast_if_needed(IRBuilder, new_val, elem_ty, "atomg_new")

            cas_result = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "seq_cst", "seq_cst", "atomg_cas")
            if op_type == "CAST":
                # See lift_atoms: ATOMG.CAST also returns the success bit, not
                # the old value. (HeCBench SASS only ever uses CAST.SPIN; CAS
                # without T is the old-value form that genuine atomic-CAS
                # idioms still expect.)
                success_i1 = IRBuilder.extract_value(cas_result, 1, "atomg_cas_succ")
                old_val = IRBuilder.zext(success_i1, elem_ty, "atomg_cas_succ_ext")
            else:
                old_val = IRBuilder.extract_value(cas_result, 0, "atomg_cas_old")
        else:
            src_val = _get_val(uses[1], "atomg_val")
            src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "atomg_val")
            
            rmw_ops = {
                "ADD": "add", "MIN": "min", "MAX": "max",
                "AND": "and", "OR": "or", "XOR": "xor",
                "EXCH": "xchg", "INC": "add", "DEC": "sub",
            }
            rmw_op = rmw_ops.get(op_type)
            if rmw_op is None:
                raise UnsupportedInstructionException(f"Unsupported ATOMG operation: {op_type}")
            
            if op_type in ("INC", "DEC"):
                src_val = self.ir.Constant(elem_ty, 1)
            
            old_val = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", f"atomg_{op_type.lower()}")
        
        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            dest_ty = dest_reg.get_ir_type(self)
            old_val = self._insert_type_cast_if_needed(IRBuilder, old_val, dest_ty, "atomg_result")
            self._set_val(IRBuilder, IRRegs, dest_reg, old_val)

    @lift_for("REDG")
    def lift_redg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        # SM90: REDG.E.ADD desc[URx][Ry.64], Rsrc
        # For x86, descriptor register is ignored — Ry.64 is the address.
        uses = Inst.get_uses()
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]

        is_64bit = "64" in modifiers
        op_type = None
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "INC", "DEC"]:
                op_type = mod
                break
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown REDG operation: {Inst.opcodes}")

        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"No memory operand found in REDG: {Inst.opcodes}")
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "redg_addr")

        is_s32 = "S32" in modifiers
        # FLOAT reductions (REDG.E.ADD.{F16,F16x2,BF16,BF16x2,F32,F64}) must use
        # atomic_rmw fadd on a FLOAT-typed pointer — emitting integer `atomicrmw
        # add` on the raw float bits is only coincidentally correct for a single
        # accumulation from 0.0 (bits+0 == bits); it adds bit-patterns (wrong) for
        # any contended/multi-add reduction. The packed cases (F16x2/BF16x2) are
        # what real stream-K GEMM epilogues actually emit (cutlass example 47 has
        # 64 REDG.E.ADD.F16x2 sites accumulating split-K partials); cuDNN pooling
        # backward emits REDG.E.ADD.F32. Detect the float type by its modifier
        # token rather than substring membership — `"F16" in modifiers` is False
        # for the token "F16X2", which previously dropped the packed-half stream-K
        # reductions into the integer path. The bug is latent under the
        # sequential-CTA wrapper (one writer at a time) but corrupts results once
        # CTAs run concurrently (CULIFTER_CONCURRENT_CTA), and corrupts any
        # window-overlap (multi-contributor) reduction even sequentially.
        is_packed_half = "F16X2" in modifiers or "BF16X2" in modifiers
        is_bf = "BF16X2" in modifiers or "BF16" in modifiers
        # Vectorized F32 reductions: REDG.E.ADD.F32x{2,4} (the FA3 bwd dQ_accum
        # epilogue uses F32x4 — desc[UR][R.64], one accumulator group of 4 floats
        # stored to consecutive words). The token is "F32X4"/"F32X2", NOT "F32", so
        # detect those explicitly (same class of bug the F16X2 comment warns about:
        # the substring "F32" is False for token "F32X4", so without this the x4
        # reduction fell into the integer `atomicrmw add` path = bit-pattern adds =
        # garbage once a cell gets >1 contribution, AND only 1 of the 4 floats was
        # stored). f32_vec = number of consecutive f32 words this REDG writes.
        f32_vec = 4 if "F32X4" in modifiers else (2 if "F32X2" in modifiers else 1)
        is_float = op_type == "ADD" and (
            is_packed_half or "F32" in modifiers or f32_vec > 1 or "F16" in modifiers
            or "BF16" in modifiers or "F64" in modifiers
        )
        if is_float:
            if is_packed_half:
                elem_ty = self.ir.VectorType(self._half2_elem_ty(is_bf), 2)
            elif "F64" in modifiers:
                elem_ty = self.ir.DoubleType()
            elif "F16" in modifiers or "BF16" in modifiers:
                elem_ty = self._half2_elem_ty(is_bf)
            else:
                elem_ty = self.ir.FloatType()
        else:
            elem_ty = self.ir.IntType(64) if is_64bit else self.ir.IntType(32)
        if is_float and f32_vec > 1:
            # REDG.E.ADD.F32x{2,4}: the source is a group of f32_vec consecutive
            # registers (R, R+1, ...) stored to consecutive words addr, addr+4, ...
            # Read each register and emit a float fadd at the right element offset.
            from sir.operand import Operand
            src_grp = Operand.create_reg_group(uses[-1], f32_vec)
            regs = src_grp.reg_group if src_grp.is_reg_group else [uses[-1]] * f32_vec
            f32 = self.ir.FloatType()
            i64 = self.ir.IntType(64)
            base_i64 = addr_val
            if base_i64.type != i64:
                base_i64 = IRBuilder.zext(base_i64, i64, "redg_vec_base") \
                    if getattr(base_i64.type, "width", 64) < 64 else base_i64
            for vi in range(f32_vec):
                v = _get_val(regs[vi], f"redg_vval{vi}")
                v = self._insert_type_cast_if_needed(IRBuilder, v, f32, f"redg_vval{vi}_f")
                eaddr = IRBuilder.add(base_i64, self.ir.Constant(i64, 4 * vi),
                                      f"redg_vaddr{vi}") if vi else base_i64
                if os.environ.get("SLIFTER_REDG_DUMP"):
                    self.run_intrinsic("slifter_redg_dump", IRBuilder,
                                       [eaddr, v], f"redgd{vi}")
                vptr = _as_pointer(eaddr, f32, f"redg_vptr{vi}")
                IRBuilder.atomic_rmw("fadd", vptr, v, "seq_cst", f"redg_vfadd{vi}")
            return

        ptr = _as_pointer(addr_val, elem_ty, "redg_ptr")

        src_val = _get_val(uses[-1], "redg_val")
        src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "redg_val")

        if is_float:
            IRBuilder.atomic_rmw("fadd", ptr, src_val, "seq_cst", "redg_fadd")
            return

        if op_type == "INC":
            # NVIDIA INC: result = (old >= val) ? 0 : old + 1
            # REDG is fire-and-forget (no return), decompose as load-compute-store.
            old_val = IRBuilder.load(ptr, "redg_inc_old")
            one = self.ir.Constant(elem_ty, 1)
            inc_val = IRBuilder.add(old_val, one, "redg_inc_add")
            zero = self.ir.Constant(elem_ty, 0)
            cmp = IRBuilder.icmp_unsigned(">=", old_val, src_val, "redg_inc_cmp")
            new_val = IRBuilder.select(cmp, zero, inc_val, "redg_inc_sel")
            IRBuilder.store(new_val, ptr)
        elif op_type == "DEC":
            # NVIDIA DEC: result = (old == 0 || old > val) ? val : old - 1
            old_val = IRBuilder.load(ptr, "redg_dec_old")
            one = self.ir.Constant(elem_ty, 1)
            dec_val = IRBuilder.sub(old_val, one, "redg_dec_sub")
            zero = self.ir.Constant(elem_ty, 0)
            is_zero = IRBuilder.icmp_unsigned("==", old_val, zero, "redg_dec_z")
            is_gt = IRBuilder.icmp_unsigned(">", old_val, src_val, "redg_dec_gt")
            wrap = IRBuilder.or_(is_zero, is_gt, "redg_dec_wrap")
            new_val = IRBuilder.select(wrap, src_val, dec_val, "redg_dec_sel")
            IRBuilder.store(new_val, ptr)
        else:
            rmw_ops = {"ADD": "add", "MIN": "min", "MAX": "max",
                       "AND": "and", "OR": "or", "XOR": "xor"}
            rmw_op = rmw_ops[op_type]
            IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", f"redg_{op_type.lower()}")

    @lift_for("RED")
    def lift_red(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        # SM80 reduce (fire-and-forget atomic). RED.E.<OP>[.<TYPE>].STRONG.<SCOPE>
        # The base Lifter.lift_red handed the raw SASS modifier to
        # IRBuilder.atomic_rmw, which breaks on uppercase mnemonics and mismatched
        # value/pointer element types. Mirror lift_redg for shape recovery.
        uses = Inst.get_uses()
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]

        is_64bit = "64" in modifiers or "U64" in modifiers or "S64" in modifiers
        is_f64 = "F64" in modifiers
        is_f32 = "F32" in modifiers
        # Packed 16-bit float pairs (F16x2/BF16x2) are 32-bit-wide vector adds,
        # NOT scalar half — using HalfType here would add only the low lane.
        is_packed_half = "F16X2" in modifiers or "BF16X2" in modifiers
        is_bf = "BF16X2" in modifiers or "BF16" in modifiers
        is_scalar_half = "F16" in modifiers or "BF16" in modifiers
        is_f16 = is_packed_half or is_scalar_half

        op_type = None
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "INC", "DEC"]:
                op_type = mod
                break
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown RED operation: {Inst.opcodes}")

        if is_f64:
            elem_ty = self.ir.DoubleType()
        elif is_f32:
            elem_ty = self.ir.FloatType()
        elif is_packed_half:
            elem_ty = self.ir.VectorType(self._half2_elem_ty(is_bf), 2)
        elif is_scalar_half:
            elem_ty = self._half2_elem_ty(is_bf)
        else:
            elem_ty = self.ir.IntType(64) if is_64bit else self.ir.IntType(32)

        mem_ops = [op for op in uses if op.is_mem_addr]
        addr_op = mem_ops[0] if mem_ops else uses[0]
        addr_val = _get_val(addr_op, "red_addr")
        ptr = _as_pointer(addr_val, elem_ty, "red_ptr")

        # NVIDIA INC/DEC aren't plain add/sub — decompose to load-op-store.
        # Float RED.ADD does not use INC/DEC, so gate on integer path only.
        if op_type in ("INC", "DEC") and not (is_f32 or is_f64 or is_f16):
            src_val = _get_val(uses[-1], f"red_{op_type.lower()}_src")
            src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, f"red_{op_type.lower()}_src")
            one = self.ir.Constant(elem_ty, 1)
            zero = self.ir.Constant(elem_ty, 0)
            if op_type == "INC":
                old_val = IRBuilder.load(ptr, "red_inc_old")
                inc_val = IRBuilder.add(old_val, one, "red_inc_add")
                cmp = IRBuilder.icmp_unsigned(">=", old_val, src_val, "red_inc_cmp")
                new_val = IRBuilder.select(cmp, zero, inc_val, "red_inc_sel")
                IRBuilder.store(new_val, ptr)
            else:  # DEC
                old_val = IRBuilder.load(ptr, "red_dec_old")
                dec_val = IRBuilder.sub(old_val, one, "red_dec_sub")
                is_zero = IRBuilder.icmp_unsigned("==", old_val, zero, "red_dec_z")
                is_gt = IRBuilder.icmp_unsigned(">", old_val, src_val, "red_dec_gt")
                wrap = IRBuilder.or_(is_zero, is_gt, "red_dec_wrap")
                new_val = IRBuilder.select(wrap, src_val, dec_val, "red_dec_sel")
                IRBuilder.store(new_val, ptr)
            return

        src_val = _get_val(uses[-1], "red_val")
        src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "red_val")

        if is_f64 or is_f32 or is_f16:
            if op_type != "ADD":
                raise UnsupportedInstructionException(
                    f"Unsupported RED float op {op_type}: {Inst.opcodes}"
                )
            rmw_op = "fadd"
        else:
            rmw_ops = {"ADD": "add", "MIN": "min", "MAX": "max",
                       "AND": "and", "OR": "or", "XOR": "xor"}
            rmw_op = rmw_ops[op_type]

        IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", f"red_{op_type.lower()}")

    @lift_for("ATOMS")
    def lift_atoms(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        
        is_64bit = "64" in modifiers
        op_type = None
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "EXCH", "CAS", "CAST", "INC", "DEC"]:
                op_type = mod
                break
        
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown ATOMS operation: {Inst.opcodes}")
        
        dest_reg = None
        for d in defs:
            if d.is_writable_reg and not d.is_pt:
                dest_reg = d
                break
        
        ptr_op = uses[0]
        addr_val = _get_val(ptr_op, "atoms_addr")
        
        elem_ty = self.ir.IntType(64) if is_64bit else self.ir.IntType(32)
        
        shared_addr = IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(self.ir.IntType(32), 0), addr_val],
            "atoms_shared_addr"
        )
        ptr = IRBuilder.bitcast(shared_addr, self.ir.PointerType(elem_ty), "atoms_ptr")

        # Shared-memory atomics participate in the lockstep turnstile just
        # like LDS/STS: ptxas emits no WARPSYNC between a warp's shared
        # zero-init STS loop and its ATOMS.POPC.INC increments (cub match-rank
        # histograms) — ordering relies on physical warp lockstep. Without the
        # bracket a leading lane's increment can land before a trailing lane's
        # zero store and be wiped out.
        self._turn_begin(IRBuilder)

        if op_type in ("CAS", "CAST"):
            cmp_val = _get_val(uses[1], "atoms_cmp")
            new_val = _get_val(uses[2], "atoms_new")

            cmp_val = self._insert_type_cast_if_needed(IRBuilder, cmp_val, elem_ty, "atoms_cmp")
            new_val = self._insert_type_cast_if_needed(IRBuilder, new_val, elem_ty, "atoms_new")

            cas_result = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "seq_cst", "seq_cst", "atoms_cas")
            if op_type == "CAST":
                # SASS ATOMS.CAST returns the success bit (1 if swap happened,
                # else 0) in the destination, not the prior value. ptxas's
                # software-CAS spin loop tests `R == 1` to decide whether to
                # retry, so giving it the old value causes the SW loop to spin
                # forever (or, on x86 single cmpxchg, exit on the wrong
                # condition). Extract the i1 success bit and zext to the dest
                # width.
                success_i1 = IRBuilder.extract_value(cas_result, 1, "atoms_cas_succ")
                old_val = IRBuilder.zext(success_i1, elem_ty, "atoms_cas_succ_ext")
            else:
                old_val = IRBuilder.extract_value(cas_result, 0, "atoms_cas_old")
        else:
            rmw_ops = {
                "ADD": "add", "MIN": "min", "MAX": "max",
                "AND": "and", "OR": "or", "XOR": "xor",
                "EXCH": "xchg", "INC": "add", "DEC": "sub",
            }
            rmw_op = rmw_ops.get(op_type)
            if rmw_op is None:
                raise UnsupportedInstructionException(f"Unsupported ATOMS operation: {op_type}")

            if op_type in ("INC", "DEC"):
                src_val = self.ir.Constant(elem_ty, 1)
            else:
                src_val = _get_val(uses[1], "atoms_val")
                src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "atoms_val")
            
            old_val = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "seq_cst", f"atoms_{op_type.lower()}")

        self._turn_end(IRBuilder)

        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            dest_ty = dest_reg.get_ir_type(self)
            old_val = self._insert_type_cast_if_needed(IRBuilder, old_val, dest_ty, "atoms_result")
            self._set_val(IRBuilder, IRRegs, dest_reg, old_val)

    @lift_for("QSPC")
    def lift_qspc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """QSPC.E.<S|G|L|C> Pd, Rd, [Ra] — generic address-space query.
        CPU backend: is_<space>_address(i64) classifies by our memory model
        (shared/local are small offsets into the wrapper arrays, global are
        host heap pointers). The base lifter passes an i8* for the nvvm
        intrinsic; our runtime takes the raw i64.
        """
        dest = Inst.get_defs()[0]
        addr_op = Inst.get_uses()[-1]
        addr_val = _get_val(addr_op, "qspc_addr")
        i64 = self.ir.IntType(64)
        addr_val = self._insert_type_cast_if_needed(IRBuilder, addr_val, i64, "qspc_addr_i64")
        space_map = {"S": "isspacep_shared", "G": "isspacep_global",
                     "L": "isspacep_local", "C": "isspacep_const"}
        space = Inst.opcodes[2]
        res = self.run_intrinsic(space_map[space], IRBuilder, [addr_val], f"qspc_{space.lower()}")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("SSY", "BSSY")
    def lift_ssy(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # Divergence-region open: record in the per-lane history (site id =
        # SASS address). NOP semantically; powers activemask reconstruction.
        site = self.ir.Constant(self.ir.IntType(32), int(Inst.id.split('_')[0], 16))
        self.run_intrinsic("warp_region_push", IRBuilder, [site], "bssy_evt")

    @lift_for("SYNC", "BSYNC")
    def lift_sync(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        site = self.ir.Constant(self.ir.IntType(32), int(Inst.id.split('_')[0], 16))
        self.run_intrinsic("warp_region_pop", IRBuilder, [site], "bsync_evt")

    @lift_for("ELECT")
    def lift_elect(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """ELECT P, Rd, PT — elect one leader lane from the converged warp.

        HW elect.sync makes exactly ONE lane (the lowest active) the leader: its
        predicate dest is true and the register dest receives the leader's lane
        id. The x86 wrapper runs every warp lane concurrently, so this is a real
        election (runtime elect_one over the history-matched group), not an
        unconditional "lane 0 elected" — otherwise every lane passes the typical
        `@!P EXIT` guard and races the leader-only store (ISA2a_elect_one wrote a
        nondeterministic tid / nothing instead of the leader's). elect_one
        returns the leader lane id; the predicate is (myLane == leader).
        """
        defs = Inst.get_defs()
        site = self._collective_site(Inst)
        leader = self.run_intrinsic("elect_one", IRBuilder, [site], "elect_leader")
        my_lane = self._emit_lane_id_i32(IRBuilder)
        is_leader = IRBuilder.icmp_unsigned("==", my_lane, leader, "elect_is_leader")
        for d in defs:
            if d.is_predicate_reg and not d.is_pt:
                self._set_val(IRBuilder, IRRegs, d, is_leader)
            elif d.is_writable_reg and not d.is_rz:
                # Register destination receives the elected leader's lane id.
                dest_ty = d.get_ir_type(self)
                lv = self._insert_type_cast_if_needed(
                    IRBuilder, leader, dest_ty, "elect_lane")
                self._set_val(IRBuilder, IRRegs, d, lv)

    @lift_for("USETMAXREG")
    def lift_usetmaxreg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """USETMAXREG.{TRY_ALLOC|DEALLOC}.CTAPOOL — Hopper warpgroup register
        reallocation (PTX setmaxnreg). On HW a warpgroup grows/shrinks its
        register allocation from the CTA register pool; on x86 there is no
        physical register pool, so the (de)allocation itself is a NOP.

        CRUCIAL: the TRY_ALLOC form writes a success PREDICATE (e.g.
        `USETMAXREG.TRY_ALLOC.CTAPOOL UP0, 0xf0`) that the kernel branches on
        with a retry loop (`@!UP0 BRA try_again`). On x86 the allocation always
        succeeds, so the predicate must be set TRUE. Leaving it unset — which is
        what the generic 'unsupported instruction' stub did — makes every
        consumer warpgroup spin forever in that retry loop (the FlashAttention-3
        warp-specialized consumer livelock: 2 warpgroups busy-spinning while the
        producer starves). DEALLOC has no predicate dest and is a plain NOP.
        """
        modifiers = [t.upper() for t in Inst.opcodes[1:]]
        if "TRY_ALLOC" in modifiers:
            true_pred = self.ir.Constant(self.ir.IntType(1), 1)
            for d in Inst.get_defs():
                if d.is_predicate_reg and not d.is_pt:
                    self._set_val(IRBuilder, IRRegs, d, true_pred)
        # DEALLOC / any other form: NOP (no register pool to manage on x86).

    @lift_for("SYNCS")
    def lift_syncs(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """SYNCS Hopper mbarrier op — lifted to the wrapper's real mbarrier
        runtime, like any other instruction (no env toggle, no special-casing).

        Variants:
          EXCH(.64)            -> mbarrier.init    -> slifter_mbar_init(off, iv)
          ARRIVE(.TRANS64...)  -> mbarrier.arrive  -> slifter_mbar_arrive(off, 0)
          PHASECHK(.TRYWAIT)   -> mbarrier.try_wait -> slifter_mbar_try_wait(off,par)
                                  (writes the readiness predicate; the SASS spins)
          CCTL.IV / RLSE / other: NOP (cache-control / release hints).

        The first memory operand is the shared-memory mbarrier address (a byte
        offset into @shared_mem); the runtime keys barrier state by (CTA, offset).
        A real barrier is required because the wrapper runs a CTA's threads
        concurrently, so a consumer must actually block on a producer's arrivals
        (a warp-specialized FA3 ring buffer is otherwise a data race). Kernels
        that use no SYNCS emit no calls and are unaffected.
        """
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)

        if not uses:
            return

        def _as_i32(v, nm):
            if isinstance(v.type, self.ir.PointerType):
                return IRBuilder.ptrtoint(v, i32, nm)
            if hasattr(v.type, "width"):
                if v.type.width > 32:
                    return IRBuilder.trunc(v, i32, nm)
                if v.type.width < 32:
                    return IRBuilder.zext(v, i32, nm)
            return v

        def _mbar_off():
            # Effective shared-memory address of the mbarrier, used as the barrier
            # key. ptxas emits it as [Rbase+URindex(+imm)] (the uniform UR carries
            # the smem region base). _get_val already adds the immediate offset for
            # a mem-addr operand, and the MemIndexFold pass has already folded the
            # index register into the base (so op.index_reg is None by lift time) —
            # so _get_val(op) is the full effective address. Adding the index or
            # offset here too would DOUBLE-count them: a [UR+0x8] barrier resolved
            # to +0x10, colliding the 8-apart barrier array and dead-locking a
            # multi-stage producer/consumer ring (ws_attn_pipe NSTAGE=2).
            #
            # Cross-register unification (FA3 bwd), RESOLVED upstream of this
            # function: ptxas builds the SAME logical barrier with DIFFERENT base
            # registers on arrive vs wait — the init/arrive path maps the barrier
            # window via `UPRMT UR4,UR5,0x654,0x400` (the DSMEM mapa base) while
            # the consumer wait uses the raw smem base ([UR6+0x38800]). On the
            # single-CTA x86 model lift_prmt (lift/lifter.py) collapses the mapa
            # base `PRMT rank,0x654,window` to 0 (gated _cluster_is_local +
            # _has_ublkcp), so BOTH sides resolve to the same local smem offset
            # and no special keying is needed here. (Keying on offset_value alone
            # was tried and COLLIDES the ws_attn_pipe multi-ring barriers.)
            return _as_i32(_get_val(uses[0], "mbar_addr"), "mbar_addr32")

        def _widen(v, ty, nm):
            if isinstance(v.type, self.ir.PointerType):
                return IRBuilder.ptrtoint(v, ty, nm)
            if hasattr(v.type, "width"):
                if v.type.width < ty.width:
                    return IRBuilder.zext(v, ty, nm)
                if v.type.width > ty.width:
                    return IRBuilder.trunc(v, ty, nm)
            return v

        if "EXCH" in modifiers:                       # mbarrier.init
            off = _mbar_off()
            iv = (_widen(_get_val(uses[1], "mbar_iv"), i64, "mbar_iv64")
                  if len(uses) > 1 else self.ir.Constant(i64, 0))
            self.run_intrinsic("slifter_mbar_init", IRBuilder, [off, iv], "mbar_init")
            return

        if "ARRIVE" in modifiers:                     # mbarrier.arrive[.expect_tx]
            off = _mbar_off()
            # Distinguish a plain count-arrive from an EXPECT-TX arrive by the
            # transaction-bytes operand (uses[1]). FA3 producers emit
            #   SYNCS.ARRIVE.TRANS64 RZ, [bar], R<bytes>   (expect-tx, non-RZ)
            # and then issue the bulk copies (UTMALDG/UBLKCP) that fill the tile;
            # on HW the phase flips only once those bytes land. A plain
            #   SYNCS.ARRIVE.TRANS64.A1T0 RZ, [bar], RZ    (ws_attn_pipe ring)
            # flips on arrival count. Route the expect-tx form to the DEFERRED-flip
            # arrive (slifter_mbar_arrive_tx): it counts the arrival but defers the
            # phase flip until the fiber's next yield, by which point the producer's
            # synchronous copies have published the tile to smem. This fixes the
            # FA3 bwd LSE/dPsum "reads zero" race without touching the count-arrive
            # path (ws_attn_pipe stays byte-identical).
            # The deferred-flip path matters only for kernels that publish a 1D
            # tile via UBLKCP between the arrive and the consumer's read (FA3 bwd
            # LSE/dPsum). Forward kernels have no UBLKCP and order correctly with
            # the eager arrive, so gating on self._has_ublkcp keeps their lifted IR
            # byte-identical (no arrive_tx emitted). General: keyed on the kernel
            # actually using UBLKCP, not on a kernel name.
            tx_op = uses[1] if len(uses) > 1 else None
            # fp8 (e4m3/e5m2) Vt-transpose-done arrive — a PLAIN COUNT arrive that
            # must defer its phase flip to the producer fiber's next yield. In the
            # warp-specialized fp8 fwd kernel, warpgroup 0 re-publishes V (MN->K
            # transposed) into a 2-stage smem ring via a warp-collective STSM and
            # signals each stage ready with `SYNCS.ARRIVE.TRANS64.A1T0 [bar], RZ`
            # (RZ tx-bytes = no expect-tx). The publish is a register->smem store,
            # not a bulk copy, so the byte-accounting expect-tx FIFO does NOT apply
            # — but the same ordering hazard does: on the fiber backend an eager
            # flip releases the PV consumer to read the Vt stage BEFORE the STSM
            # runs, dropping the partial last K-tile's PV (O = 224/256·v_descale).
            # Route this one form to the COUNT deferred-flip arrive so the STSM
            # publishes before the consumer observes the flip. Gated narrowly:
            #   - self._fa_fp8  (fp16/bf16 fwd + bwd never set it -> byte-identical)
            #   - "A1T0" in modifiers AND "RED" not in modifiers — uniquely the
            #     transpose-producer arrive (the .RED.A1T0 reduction-arrives and the
            #     non-A1T0 expect-tx Q/K-ready arrives keep their existing eager path,
            #     so the fp8 Q/K-ready barriers do not deadlock).
            fp8_vt_arrive = (getattr(self, "_fa_fp8", False)
                             and "A1T0" in modifiers and "RED" not in modifiers)
            expect_tx = (getattr(self, "_has_ublkcp", False)
                         and tx_op is not None and not tx_op.is_rz)
            # Arrive-count qualifier: `.A1T0` (Arrive 1, T0) contributes +1 to the
            # barrier's arrival count; `.A0T1` (Arrive 0, T1) contributes +0 — it is
            # a pure ordering/release arrive, not a count arrive. The FA3 gather
            # producers signal a data-ready (full) barrier per stage with the triple
            #   SYNCS.ARRIVE.RED.A0T1 [bar]   (count 0)
            #   ARRIVES.LDGSTSBAR     [bar]   (cp.async tx-arm, count 0 -> lift_arrives)
            #   SYNCS.ARRIVE.A1T0     [bar]   (count 1)
            # = exactly +1 net per producer thread, so the barrier (expect = warpgroup
            # size) flips ONCE when the whole warpgroup has arrived. Lifting the A0T1
            # as a +1 count-arrive double-counts each thread, so the full barrier
            # flips TWICE per phase (256 arrivals on an expect-128 barrier) — it
            # returns to its original parity and a consumer waiting the post-flip
            # parity MISSES the wakeup, blocks on the full barrier forever, never runs
            # its consumer_release, and the producer deadlocks waiting to refill the
            # never-freed empty barrier (the proven paged_split rc=124 under fibers).
            # Route A0T1 to the no-count release arrive so only A1T0 supplies the
            # count. General (HW-faithful arrive-count semantics); kernels whose
            # arrives are all A1T0 (ws_attn_pipe, the fp8 Vt ring) are unaffected.
            a0t1_nocount = ("A0T1" in modifiers)
            if a0t1_nocount:
                self.run_intrinsic("slifter_mbar_ldgsts_arrive", IRBuilder,
                                   [off], "mbar_arrive_a0t1")
            elif fp8_vt_arrive:
                self.run_intrinsic("slifter_mbar_arrive_deferred", IRBuilder,
                                   [off], "mbar_arrive_deferred")
            elif expect_tx:
                # Pass the expect-tx byte count so the flip waits until that many
                # bytes are delivered by the bulk copies (byte-accounting FIFO).
                tx_bytes = _widen(_get_val(tx_op, "mbar_tx_bytes"), i64, "mbar_tx_bytes64")
                self.run_intrinsic("slifter_mbar_arrive_tx", IRBuilder,
                                   [off, tx_bytes], "mbar_arrive_tx")
            else:
                self.run_intrinsic("slifter_mbar_arrive", IRBuilder,
                                   [off, self.ir.Constant(i64, 0)], "mbar_arrive")
            return

        if "PHASECHK" in modifiers or "TRYWAIT" in modifiers:   # mbarrier.try_wait
            off = _mbar_off()
            par = (_widen(_get_val(uses[1], "mbar_par"), i32, "mbar_par32")
                   if len(uses) > 1 else self.ir.Constant(i32, 0))
            ready = self.run_intrinsic("slifter_mbar_try_wait", IRBuilder,
                                       [off, par], "mbar_trywait")
            pred = IRBuilder.icmp_unsigned("!=", ready,
                                           self.ir.Constant(i32, 0), "mbar_ready")
            for d in defs:
                if d.is_predicate_reg and not d.is_pt:
                    self._set_val(IRBuilder, IRRegs, d, pred)
                    return
            return
        # CCTL.IV / RLSE / other: NOP.

    @lift_for("STAS")
    def lift_stas(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """STAS async store marker — NOP on x86 single-threaded."""
        pass

    @lift_for("REDAS")
    def lift_redas(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """REDAS async reduction marker — NOP on x86 single-threaded."""
        pass

    @lift_for("ARRIVES")
    def lift_arrives(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """ARRIVES.LDGSTSBAR.64.TRANSCNT [mbar] — mbarrier arrive-on-cp.async.

        The FlashAttention-forward producer handshake: `cp.async.mbarrier.arrive`
        increments the shared-memory mbarrier so a consumer's `mbarrier.try_wait`
        can detect the async copies have landed. Single memory operand (the smem
        mbarrier address); no register def.

        Arrive the real mbarrier, exactly like the SYNCS.ARRIVE sibling. The WAIT
        side is `SYNCS.PHASECHK.TRANS64.TRYWAIT`, which lift_syncs lowers to a REAL
        `slifter_mbar_try_wait` (it blocks until the producer's arrivals land — the
        wrapper runs a CTA's warps concurrently, so a warp-specialized consumer
        genuinely waits). A cp.async-only producer (packgqa / split / paged) signals
        its tile ONLY through ARRIVES; if this is a NOP the consumer's try_wait
        never flips phase and the kernel hangs immediately after the first load.
        (The earlier NOP assumed PHASECHK was an unconditional-ready predicate;
        that is no longer true — lift_syncs now does a real try_wait.) The base
        TMA-only kernels emit no ARRIVES, so they are unaffected.
        """
        uses = Inst.get_uses()
        if not uses:
            return
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        addr = _get_val(uses[0], "arrives_mbar")
        if isinstance(addr.type, self.ir.PointerType):
            off = IRBuilder.ptrtoint(addr, i32, "arrives_mbar32")
        elif hasattr(addr.type, "width") and addr.type.width != 32:
            conv = IRBuilder.trunc if addr.type.width > 32 else IRBuilder.zext
            off = conv(addr, i32, "arrives_mbar32")
        else:
            off = addr
        # `ARRIVES.LDGSTSBAR.TRANSCNT` is a cp.async transaction-completion arrive
        # that, on HW, ARMS transaction tracking but does NOT contribute a plain
        # arrival count — the count comes from the PAIRED `SYNCS.ARRIVE.A1T0` on
        # the same barrier offset (verified: every ARRIVES in packgqa/split/paged/
        # paged_split is immediately followed by a same-offset SYNCS.ARRIVE.A1T0).
        # The previous eager count-arrive double-counted each producer thread, so
        # the data-ready (full) barrier flipped after only half the producer
        # warpgroup had gathered — the proven paged_split premature-flip
        # (consumers read WG2's Q as zero -> corr 0.977). Route the gather
        # producers' ARRIVES to the no-count tx-arm (slifter_mbar_ldgsts_arrive):
        # the paired SYNCS.ARRIVE provides the count, and the synchronous LDGSTS
        # has already published its bytes, so the transaction is satisfied. Gated
        # on self._q_gather_loaded (any LDGSTS in the kernel); TMA-only kernels
        # emit no ARRIVES at all, so they are byte-identical regardless.
        if getattr(self, "_q_gather_loaded", False):
            self.run_intrinsic("slifter_mbar_ldgsts_arrive", IRBuilder,
                               [off], "arrives_ldgsts_arrive")
        else:
            self.run_intrinsic("slifter_mbar_arrive", IRBuilder,
                               [off, self.ir.Constant(i64, 0)], "arrives_arrive")

    @lift_for("NANOSLEEP")
    def lift_nanosleep(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """NANOSLEEP.SYNCS delay instruction — NOP on x86 single-threaded."""
        pass

    @lift_for("FENCE")
    def lift_fence(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """FENCE.VIEW.ASYNC — memory ordering fence, NOP on x86 single-threaded."""
        pass

    @lift_for("WARPGROUP")
    def lift_warpgroup(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """WARPGROUP.ARRIVE/DEPBAR — warpgroup synchronization, NOP on x86 single-threaded."""
        pass

    @lift_for("UBLKCP")
    def lift_ublkcp(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """UBLKCP.S.G [URdst], [URsrc], URsize — TMA bulk copy global→shared.

        cp.async.bulk.shared::cluster.global: URdst is the shared-memory
        byte offset (bits 24+ carry the cluster CTA rank — masked off, the
        x86 wrapper runs cluster-of-1), URsrc is an implicit 64-bit UR pair
        holding the global source address (expanded in sir/instruction.py),
        and URsize is the copy size in 16-byte units (FA3 ConvertdQ copies
        its 80x128 f32 tile = 40960 B with URsize = 0xa00).

        Lowered as a synchronous llvm.memcpy into shared_mem; the paired
        mbarrier protocol (SYNCS.EXCH / ARRIVE.TRANS64 / PHASECHK.TRYWAIT)
        is already always-ready on x86, so copy-then-barrier ordering holds.
        """
        i1 = self.ir.IntType(1)
        i8 = self.ir.IntType(8)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        uses = Inst.get_uses()
        dst_op, src_op, size_op = uses[0], uses[1], uses[2]

        dst_off = _get_val(dst_op, "ublkcp_dst")
        dst_off = self._insert_type_cast_if_needed(
            IRBuilder, dst_off, i32, "ublkcp_dst_i32")
        dst_off = IRBuilder.and_(
            dst_off, self.ir.Constant(i32, 0xFFFFFF), "ublkcp_dst_off")
        dst_ptr = IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(i32, 0), dst_off],
            "ublkcp_dst_ptr",
        )

        src_addr = _get_val(src_op, "ublkcp_src")  # UR pair → i64
        src_addr = self._insert_type_cast_if_needed(
            IRBuilder, src_addr, i64, "ublkcp_src_i64")
        src_ptr = IRBuilder.inttoptr(
            src_addr, self.ir.PointerType(i8), "ublkcp_src_ptr")

        size_units = _get_val(size_op, "ublkcp_units")
        size_units = self._insert_type_cast_if_needed(
            IRBuilder, size_units, i64, "ublkcp_units_i64")
        size_bytes = IRBuilder.shl(
            size_units, self.ir.Constant(i64, 4), "ublkcp_bytes")

        import os as _os
        _ublkcp_trace = (_os.environ.get("SLIFTER_UBLKCP_DUMP")
                         or _os.environ.get("SLIFTER_LSE_TRACE"))

        self.run_intrinsic(
            "memcpy_i64", IRBuilder,
            [dst_ptr, src_ptr, size_bytes, self.ir.Constant(i1, 0)])

        # DEBUG (gated, lift-time): dump the bulk-copy dest smem offset + the first
        # source f32 + byte count AFTER the memcpy, so the tracker can peek the LIVE
        # destination smem (post-write) and detect a later clobber. Only emitted
        # under env at lift time -> production lifts byte-identical.
        if _ublkcp_trace:
            self.run_intrinsic("slifter_ublkcp_dump", IRBuilder,
                               [dst_off, src_ptr, size_bytes], "ublkcp_dump")
        # The bulk copy just published `size_bytes` bytes; retire them against the
        # oldest outstanding expect-tx barrier (it flips once fully delivered — after
        # this copy ran, even if the producer yielded between the arrive and here).
        # No-op when no expect-tx is pending.
        self.run_intrinsic("slifter_mbar_deliver_tx", IRBuilder, [size_bytes], "mbar_deliver")

    @lift_for("UBLKRED")
    def lift_ublkred(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """UBLKRED.G.S.ADD.F32.RN [gmem_addr], [smem_off], size, desc[URn].

        TMA bulk REDUCE-add global += shared, fp32 — the FA3 BACKWARD dQ_accum
        accumulation (the ONLY writer of dQ_accum). Mirror of UBLKCP but reversed
        direction (gmem dst, smem src) and with a reduce-add instead of a copy:
            for each fp32 elem i in [0, size_bytes/4):
                gmem_f32[gmem_addr/4 + i] += smem_f32[(smem_off + 4i)/4]

        SASS operands (all uses; UBLKRED defines no register — see
        Instruction.use_op_start_idx):
            uses[0] = global DST address (64-bit UR pair, full address)
            uses[1] = shared SRC byte offset (32-bit)
            uses[2] = reduce size in 16-byte units (URsize)
            uses[3] = desc[URn] access descriptor (cache/format; not a base) —
                      unused here, the full gmem address is already in uses[0].
        Lowered 1:N to one ublk_reduce_add(gmem_addr, smem_base, smem_off,
        size_bytes) call; the element loop + atomic fadd (for CTA concurrency)
        live in the C++ wrapper.
        """
        i8 = self.ir.IntType(8)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        uses = Inst.get_uses()
        gmem_op, smem_op, size_op = uses[0], uses[1], uses[2]

        # uses[0] = global DST address (UR pair -> i64).
        gmem_addr = _get_val(gmem_op, "ublkred_gmem")
        if isinstance(gmem_addr.type, self.ir.PointerType):
            gmem_addr = IRBuilder.ptrtoint(gmem_addr, i64, "ublkred_gmem_i64")
        elif hasattr(gmem_addr.type, 'width'):
            if gmem_addr.type.width < 64:
                gmem_addr = IRBuilder.zext(gmem_addr, i64, "ublkred_gmem_z")
            elif gmem_addr.type.width > 64:
                gmem_addr = IRBuilder.trunc(gmem_addr, i64, "ublkred_gmem_t")

        # uses[1] = shared SRC byte offset (mask off the cluster-rank bits 24+,
        # same as UBLKCP's dst offset).
        smem_off = _get_val(smem_op, "ublkred_smem")
        smem_off = self._insert_type_cast_if_needed(
            IRBuilder, smem_off, i32, "ublkred_smem_i32")
        smem_off = IRBuilder.and_(
            smem_off, self.ir.Constant(i32, 0xFFFFFF), "ublkred_smem_off")
        smem_base = IRBuilder.bitcast(
            IRBuilder.gep(self.SharedMem,
                          [self.ir.Constant(i64, 0), self.ir.Constant(i64, 0)],
                          "ublkred_smem_base_gep"),
            self.ir.PointerType(i8), "ublkred_smem_base")

        # uses[2] = reduce size in 16-byte units -> bytes.
        size_units = _get_val(size_op, "ublkred_units")
        size_units = self._insert_type_cast_if_needed(
            IRBuilder, size_units, i64, "ublkred_units_i64")
        size_bytes = IRBuilder.shl(
            size_units, self.ir.Constant(i64, 4), "ublkred_bytes")

        self.run_intrinsic("ublk_reduce_add", IRBuilder,
                           [gmem_addr, smem_base, smem_off, size_bytes])

    @lift_for("UTMALDG")
    def lift_utmaldg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """UTMALDG.4D — TMA bulk load global→shared memory.

        Phase 3 implementation: decompose to element-wise memcpy using
        descriptor base address from UR register pair.

        SASS: UTMALDG.4D [URdst], [URcoord], desc[URdesc]
        Parsed as: defs[0]=[URdst], uses[0]=[URcoord], uses[1]=desc[URdesc]

        The descriptor register (URdesc) was loaded by ULDC.64 from const_mem,
        and contains a 64-bit pointer to a CUtensorMap in const bank 0 or 4.
        qword[0] of CUtensorMap = global base address.

        On x86 single-thread, we resolve the descriptor base address and use
        the coordinate register as a byte offset into the global tensor.
        The tile is copied linearly to shared memory starting at offset 0.

        NOTE: This is a simplified decomposition. Full correctness requires:
        - Coordinate-to-address mapping using descriptor dims/strides
        - Swizzle pattern application during shared memory write
        - Correct shared memory offset from mbarrier address
        For now, we do a linear copy from base+coord_offset to shared memory.
        """
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        i8 = self.ir.IntType(8)

        # UTMALDG comes in two SASS shapes, both of which we lower to one runtime
        # call into the x86 wrapper (utma_load) that performs the genuine coord ->
        # swizzled-shared-memory tile copy. The 1:N philosophy holds: one SASS op
        # -> one runtime call; the layout/swizzle logic lives in the wrapper
        # (testable C++), not the backend. Because UTMALDG defines no register
        # (it writes only shared memory), the parser classifies EVERY operand as
        # a use (see Instruction.use_op_start_idx) — so the smem-dest ADDRESS
        # stays linked to its UPRMT/UIADD build chain instead of being handed a
        # disconnected SSA def.
        #
        #   2D form:  UTMALDG.2D [smem_dst], [desc_addr]
        #             uses[0] = [smem_dst]  (smem byte offset)
        #             uses[1] = [desc_addr] (address of the CUtensorMap-shaped
        #                                    descriptor the host driver built)
        #   4D form:  UTMALDG.4D [smem_dst], [coord_addr], desc[URn]
        #             uses[0] = [smem_dst]
        #             uses[1] = [coord_addr] (ptr to packed i32 tensor coords)
        #             uses[2]/desc[URn] = descriptor (const-bank offset/ptr)
        #
        # The descriptor the wrapper reads is the host-materialized struct:
        #   struct SlifterTmaDesc {
        #     void*   global_base;   // qword 0
        #     int32_t rows;          // tile rows (M or N)
        #     int32_t cols_k;        // tile K extent
        #     int32_t elem_bytes;    // 2 for fp16/bf16
        #     int32_t swizzle;       // 0=none(K_INTER),1=SW32,2=SW64,3=SW128
        #   };
        # global coords come from the coord operand (4D) or are 0 (2D single tile).
        if not uses:
            return

        def _to_i64(v, nm):
            if isinstance(v.type, self.ir.PointerType):
                return IRBuilder.ptrtoint(v, i64, nm)
            if hasattr(v.type, 'width'):
                if v.type.width < 64:
                    return IRBuilder.zext(v, i64, nm)
                if v.type.width > 64:
                    return IRBuilder.trunc(v, i64, nm)
            return v

        # Shared-memory destination byte offset (uses[0]; the reg holds the smem
        # byte offset — on x86 the shared window is based at 0).
        smem_off = _get_val(uses[0], "utma_smem_off")
        if isinstance(smem_off.type, self.ir.PointerType):
            smem_off = IRBuilder.ptrtoint(smem_off, i32, "utma_smem_off_i")
        elif hasattr(smem_off.type, 'width'):
            if smem_off.type.width > 32:
                smem_off = IRBuilder.trunc(smem_off, i32, "utma_smem_off32")
            elif smem_off.type.width < 32:
                smem_off = IRBuilder.zext(smem_off, i32, "utma_smem_off32")

        # Operand roles (verified against the cute SM90_TMA_LOAD PTX,
        # `cp.async.bulk.tensor.Nd [smem], [DESC, {coords}], [mbar] [, cache_hint]`):
        #
        #   2D microbench form:  UTMALDG.2D  [smem], [DESC]
        #       uses[0]=smem, uses[1]=DESC (= uses[-1])
        #   FA 4D cache-hint form: UTMALDG.4D [smem], [DESC], desc[hint]
        #       uses[0]=smem, uses[1]=DESC, uses[2]=L2 cache_hint
        #
        # The descriptor (CUtensorMap pointer) is ALWAYS uses[1] — the `[URx]`
        # operand right after the smem dest. The trailing `desc[URn]` in the FA
        # form is the .L2::cache_hint policy (a baked `UMOV`-immediate like
        # 0x14f00000_00000000), NOT the tensor map — a prior diagnosis mistook it
        # for a grid_constant-CUtensorMap offset and read desc_addr from it,
        # giving 0/garbage. uses[1] resolves to a real const-bank pointer
        # (FA: ULDC.64 c[0x0][0x198] = Params base, +0x2b0 = K descriptor).
        # The descriptor operand is the first non-coord use after smem (uses[1]).
        # TmaCoordRecover appends the 4 tensor coords as tagged use operands at
        # the END, so the descriptor at uses[1] is unaffected.
        desc_addr = _to_i64(_get_val(uses[1], "utma_desc_addr"), "utma_desc64")

        # 4D tensor coordinates {c0,c1,c2,c3}. The cute SM90_TMA_LOAD coords are
        # NOT printed operands / encoded in the instr word — they ride an implicit
        # uniform-register window (UR10..UR13). TmaCoordRecover (a pre-SSA pass)
        # attaches them as tagged use operands recovered by reaching-def, so each
        # UTMALDG.4D carries ITS OWN per-tile coords (e.g. the seq-tile row that
        # steps each loop iteration). Without them every tile loaded at (0,0) and
        # FA was numerically wrong. 2D microbench forms have no coord operands →
        # coords default to 0 (single tile encoded in the descriptor).
        # ONLY TmaCoordRecover's is_tma_coord operands are usable here: they
        # carry the 4-coord convention the wrapper API (utma_load c0..c3,
        # SlifterTmaDesc stride_c2/c3) was calibrated against. The parser's
        # synthesized tuple members (tma_tuple_idx >= 2) must NOT be used as
        # a fallback — their raw tensormap-member order mismatches the
        # 4-coord slots (bwd 5D store: c3=10240), and for single-tile 2D
        # microbenches (tma_hgmma) where the recover pass rightly finds
        # nothing, the correct coords are the 0 defaults, not the members.
        coord_ops = [op for op in uses if getattr(op, "is_tma_coord", False)]
        coord_vals = []
        for ci in range(4):
            if ci < len(coord_ops):
                v = _get_val(coord_ops[ci], f"utma_coord{ci}")
                if isinstance(v.type, self.ir.PointerType):
                    v = IRBuilder.ptrtoint(v, i32, f"utma_coord{ci}_i")
                elif hasattr(v.type, 'width'):
                    if v.type.width > 32:
                        v = IRBuilder.trunc(v, i32, f"utma_coord{ci}_t")
                    elif v.type.width < 32:
                        v = IRBuilder.zext(v, i32, f"utma_coord{ci}_z")
                coord_vals.append(v)
            else:
                coord_vals.append(self.ir.Constant(i32, 0))

        # Base pointer of this CTA's shared_mem segment (i8*), so the wrapper
        # writes the tile at smem_base + smem_off without knowing our smem layout.
        smem_base = IRBuilder.bitcast(
            IRBuilder.gep(self.SharedMem,
                          [self.ir.Constant(i64, 0), self.ir.Constant(i64, 0)],
                          "utma_smem_base_gep"),
            self.ir.PointerType(i8), "utma_smem_base")

        # One runtime call performs the genuine coord->swizzled-smem tile copy.
        # Signature: utma_load(desc, smem_base, smem_off, c0, c1, c2, c3).
        self.run_intrinsic("utma_load", IRBuilder,
                           [desc_addr, smem_base, smem_off] + coord_vals,
                           "utma_load")
        # The bulk-load byte delivery is reported from inside utma_load (the wrapper
        # knows the tile's exact byte count); doing it here would require duplicating
        # that arithmetic. utma_load calls slifter_mbar_deliver_tx itself (no-op when
        # nothing is pending -> forward byte-identical).

    @lift_for("UTMASTG")
    def lift_utmastg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """UTMASTG.Nd — TMA bulk store shared→global memory (FA epilogue O write).

        SASS: UTMASTG.Nd [smem_src], [desc]  (ptxas emits the SASS operands in
        the SAME positional order as UTMALDG even though the cute PTX text lists
        the descriptor first). The parser widens operand[-1] (the descriptor) to
        its URn:URn+1 pair, so:
            uses[0]  = shared-memory SOURCE byte offset
            uses[-1] = descriptor (64-bit O CUtensorMap pointer pair)
            tagged is_tma_coord operands = the Nd tile coordinates (UR window),
                    recovered by TmaCoordRecover when present (else coord=0).
        Lowered 1:N to one `utma_store(desc, smem_base, smem_off, c0..c3)` call;
        the smem-layout/swizzle and global addressing live in the C++ wrapper.
        """
        uses = Inst.get_uses()
        if len(uses) < 2:
            return
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        i8 = self.ir.IntType(8)

        def _to_i64(v, nm):
            if isinstance(v.type, self.ir.PointerType):
                return IRBuilder.ptrtoint(v, i64, nm)
            if hasattr(v.type, 'width'):
                if v.type.width < 64:
                    return IRBuilder.zext(v, i64, nm)
                if v.type.width > 64:
                    return IRBuilder.trunc(v, i64, nm)
            return v

        # Descriptor (CUtensorMap pointer): under the post-64a0f1a4 parse the
        # roles are [URt tuple base], [URd tensormap], {synthesized tuple
        # members, tma_tuple_idx-tagged}, optional desc[URm] access-policy
        # memdesc (tma_memdesc-tagged). Never pick a tuple member (its value
        # can be a small coord like 1 — that made the FA3 bwd epilogue call
        # utma_store(desc_addr=1) -> segfault) or the memdesc. Pre-64a0f1a4
        # lifts fall back to the old last-non-coord rule.
        desc_op = next(
            (op for op in uses[1:]
             if getattr(op, "is_mem_addr", False)
             and not getattr(op, "tma_memdesc", False)
             and getattr(op, "tma_tuple_idx", None) is None
             and not getattr(op, "is_tma_coord", False)),
            None)
        if desc_op is None:
            non_coord = [op for op in uses if not getattr(op, "is_tma_coord", False)]
            desc_op = non_coord[-1] if non_coord else uses[-1]
        desc_addr = _to_i64(_get_val(desc_op, "utmast_desc_addr"), "utmast_desc64")

        # uses[0] = shared-memory SOURCE byte offset.
        smem_off = _get_val(uses[0], "utmast_smem_off")
        if isinstance(smem_off.type, self.ir.PointerType):
            smem_off = IRBuilder.ptrtoint(smem_off, i32, "utmast_smem_off_i")
        elif hasattr(smem_off.type, 'width'):
            if smem_off.type.width > 32:
                smem_off = IRBuilder.trunc(smem_off, i32, "utmast_smem_off32")
            elif smem_off.type.width < 32:
                smem_off = IRBuilder.zext(smem_off, i32, "utmast_smem_off32")

        # Coordinates: ONLY TmaCoordRecover's is_tma_coord operands (see
        # lift_utmaldg's coord note: the parser's synthesized tuple members
        # mismatch the 4-coord slots and break zero-coord single-tile forms).
        coord_ops = [op for op in uses if getattr(op, "is_tma_coord", False)]
        coord_vals = []
        for ci in range(4):
            if ci < len(coord_ops):
                v = _get_val(coord_ops[ci], f"utmast_coord{ci}")
                if isinstance(v.type, self.ir.PointerType):
                    v = IRBuilder.ptrtoint(v, i32, f"utmast_coord{ci}_i")
                elif hasattr(v.type, 'width'):
                    if v.type.width > 32:
                        v = IRBuilder.trunc(v, i32, f"utmast_coord{ci}_t")
                    elif v.type.width < 32:
                        v = IRBuilder.zext(v, i32, f"utmast_coord{ci}_z")
                coord_vals.append(v)
            else:
                coord_vals.append(self.ir.Constant(i32, 0))

        smem_base = IRBuilder.bitcast(
            IRBuilder.gep(self.SharedMem,
                          [self.ir.Constant(i64, 0), self.ir.Constant(i64, 0)],
                          "utmast_smem_base_gep"),
            self.ir.PointerType(i8), "utmast_smem_base")

        self.run_intrinsic("utma_store", IRBuilder,
                           [desc_addr, smem_base, smem_off] + coord_vals,
                           "utma_store")

    def _shared_half_load(self, IRBuilder, byte_off_val, name):
        """Load one fp16 from @shared_mem at a runtime byte offset, return f32."""
        i32 = self.ir.IntType(32)
        half = self.ir.HalfType()
        f32 = self.ir.FloatType()
        gep = IRBuilder.gep(self.SharedMem,
                            [self.ir.Constant(i32, 0), byte_off_val], f"{name}_gep")
        hp = IRBuilder.bitcast(gep, self.ir.PointerType(half), f"{name}_hp")
        h = IRBuilder.load(hp, name, typ=half)
        return IRBuilder.fpext(h, f32, f"{name}_f32")

    def _shared_16_load(self, IRBuilder, byte_off_val, name, is_bf16=False):
        """Load one 16-bit float (fp16 or bf16) from @shared_mem at a runtime
        byte offset, return f32. fp16 path delegates to _shared_half_load so the
        existing fp16 kernels stay byte-identical; bf16 reads the 16-bit value as
        the top half of an f32 (bf16 = f32 with the low 16 mantissa bits dropped)."""
        if not is_bf16:
            return self._shared_half_load(IRBuilder, byte_off_val, name)
        i32 = self.ir.IntType(32)
        i16 = self.ir.IntType(16)
        f32 = self.ir.FloatType()
        gep = IRBuilder.gep(self.SharedMem,
                            [self.ir.Constant(i32, 0), byte_off_val], f"{name}_gep")
        bp = IRBuilder.bitcast(gep, self.ir.PointerType(i16), f"{name}_bp")
        bits16 = IRBuilder.load(bp, name, typ=i16)
        b32 = IRBuilder.shl(IRBuilder.zext(bits16, i32, f"{name}_z"),
                            self.ir.Constant(i32, 16), f"{name}_sh")
        return IRBuilder.bitcast(b32, f32, f"{name}_f32")

    def _e4m3_byte_to_f32(self, IRBuilder, byte_i32, name):
        """Decode an OCP e4m3 (FP8: 1 sign, 4 exp bias-7, 3 mantissa) byte (held
        in the low 8 bits of byte_i32) to f32, branch-free in IR. Handles zero,
        subnormals, and normals; NaN encodings (0x7f/0xff) won't appear in valid
        FA3 inputs so they map through the normal path (harmless). Matches the
        independently computed CPU e4m3-to-f32 reference."""
        i32 = self.ir.IntType(32)
        f32 = self.ir.FloatType()
        c = lambda v: self.ir.Constant(i32, v)
        u = IRBuilder.and_(byte_i32, c(0xFF), f"{name}_u")
        sign = IRBuilder.and_(IRBuilder.lshr(u, c(7), f"{name}_ss"), c(1), f"{name}_sgn")
        exp = IRBuilder.and_(IRBuilder.lshr(u, c(3), f"{name}_es"), c(0xF), f"{name}_exp")
        man = IRBuilder.and_(u, c(0x7), f"{name}_man")
        manf = IRBuilder.uitofp(man, f32, f"{name}_manf")
        eight = self.ir.Constant(f32, 8.0)
        frac = IRBuilder.fdiv(manf, eight, f"{name}_frac")     # man/8 in [0,7/8]
        # normal mant 1+man/8 ; subnormal mant man/8
        one = self.ir.Constant(f32, 1.0)
        mant_norm = IRBuilder.fadd(one, frac, f"{name}_mn")
        is_exp0 = IRBuilder.icmp_unsigned("==", exp, c(0), f"{name}_e0")
        mant = IRBuilder.select(is_exp0, frac, mant_norm, f"{name}_mant")
        # exponent: normal -> exp-7 ; subnormal -> -6. All within f32 normal
        # range ([-6, 8]) so 2^e is exact via the IEEE bit pattern (127+e)<<23
        # (avoids llvm.exp2's float approximation).
        expm = IRBuilder.sub(exp, c(7), f"{name}_em")            # exp-7
        expe = IRBuilder.select(is_exp0, c(-6), expm, f"{name}_ee")
        pbits = IRBuilder.shl(IRBuilder.add(expe, c(127), f"{name}_eb127"),
                              c(23), f"{name}_pbits")
        pow2 = IRBuilder.bitcast(pbits, f32, f"{name}_p2")
        mag = IRBuilder.fmul(mant, pow2, f"{name}_mag")
        # sign: (1 - 2*sign)
        signf = IRBuilder.sitofp(sign, f32, f"{name}_sgnf")
        two = self.ir.Constant(f32, 2.0)
        signmul = IRBuilder.fsub(one, IRBuilder.fmul(two, signf, f"{name}_2s"), f"{name}_sm")
        return IRBuilder.fmul(mag, signmul, f"{name}_v")

    def _shared_8_e4m3_load(self, IRBuilder, byte_off_val, name):
        """Load one e4m3 fp8 byte from @shared_mem at a runtime byte offset and
        decode to f32."""
        i32 = self.ir.IntType(32)
        i8 = self.ir.IntType(8)
        gep = IRBuilder.gep(self.SharedMem,
                            [self.ir.Constant(i32, 0), byte_off_val], f"{name}_gep")
        bp = IRBuilder.bitcast(gep, self.ir.PointerType(i8), f"{name}_bp")
        b8 = IRBuilder.load(bp, name, typ=i8)
        b32 = IRBuilder.zext(b8, i32, f"{name}_z")
        return self._e4m3_byte_to_f32(IRBuilder, b32, name)

    def _layout_bits_30_31(self, val):
        """Recover bits[31:30] (GMMA layout_type) of a descriptor high word that
        reaches the lifter wrapped in bitcast/trunc/zext/sext and possibly an
        add/or carry chain. The layout_type immediate (e.g. 0x40000040) is added
        to small low-word carries that cannot reach bit 30, so it suffices to
        find any constant operand in the (peeled) expression and read its bits
        [31:30]. Returns 0..3 or None."""
        seen = 0
        stack = [val]
        while stack and seen < 40:
            cur = stack.pop()
            if cur is None:
                continue
            seen += 1
            c = getattr(cur, "constant", None)
            if c is not None and isinstance(getattr(cur, "type", None), self.ir.IntType):
                try:
                    iv = int(c) & ((1 << cur.type.width) - 1)
                except Exception:
                    continue
                lt = (iv >> 30) & 0x3
                if lt != 0:          # found the layout immediate
                    return lt
                continue             # 0 may be a carry/zero operand; keep looking
            opname = getattr(cur, "opname", None) or getattr(cur, "opcode", None)
            operands = list(getattr(cur, "operands", []) or [])
            if opname in ("bitcast", "trunc", "zext", "sext", "add", "or", "and",
                          "sub", "shl") and operands:
                stack.extend(operands)
        return None

    def _peel_int_const(self, val):
        """Recover the underlying integer immediate of an IR value, peeling
        bitcast/trunc/zext/sext/and(x,allones) wrappers. Returns an int or None.
        Used to read SASS-immediate descriptor layout bits that reach the lifter
        wrapped in bitcasts (e.g. `bitcast i32 0x40000040 to float`)."""
        seen = 0
        cur = val
        while cur is not None and seen < 12:
            seen += 1
            # Direct integer constant.
            c = getattr(cur, "constant", None)
            if c is not None and isinstance(cur.type, self.ir.IntType):
                try:
                    return int(c) & ((1 << cur.type.width) - 1)
                except Exception:
                    return None
            # Float constant holding reinterpreted int bits — peel the bitcast
            # that produced it instead (handled below via operands).
            opname = getattr(cur, "opname", None) or getattr(cur, "opcode", None)
            operands = list(getattr(cur, "operands", []) or [])
            if opname in ("bitcast", "trunc", "zext", "sext") and operands:
                cur = operands[0]
                continue
            return None
        return None

    def _sw128_phys_off(self, IRBuilder, tile_base_byte, row_val, k_val, K_total, nm,
                        extra_row_swizzle=False, byte_per_elem=2, swiz_B=3,
                        compact_swizzle=False):
        """Physical byte offset of logical (row,k) in a K-major swizzled tile.

        General GMMA swizzle, parametrized by the swizzle bit-width B (swiz_B):
          ATOM = 8 << B   (B3=64, B2=32, B1=16, B0=8)   = the atom column width
          e    = row*ATOM + k   (k stays within one ATOM; the descriptor base
                                 carries the cross-atom K jump)
          yyy  = (e >> 7) & ((1<<B) - 1) ;  e ^= yyy << 4
          byte = tile_base + e*byte_per_elem
        (byte_per_elem = 2 for fp16/bf16, 1 for fp8 e4m3).

        B=3 (SW128) reproduces the original fixed `e = row*64 + k; yyy=(e>>7)&7;
        e^=yyy<<4` BYTE-FOR-BYTE (verified row<64,k<64), so every SW128 kernel
        (hd128/192/256 + all fwd) is unchanged. B=2 (SW64) / B=1 (SW32) are the
        hd96 mixed atoms, derived EXACT vs cute's full SmemLayoutQ/K/V
        (probe_full2_h96: SH=7, XP=4, rowStride=ATOM, mask=(1<<B)-1). The caller
        passes K_total = the row stride (= ATOM = 8<<swiz_B for fp16/bf16, 128 for
        fp8's wider 1-byte atom); only the swizzle MASK is taken from swiz_B here,
        so fp8 (K_total=128, swiz_B=3, mask=7) stays byte-identical. Probed
        exactly against cute for fp8 K_total=128. row/k are runtime i32 IR values.

        extra_row_swizzle: when the tile was filled by the PackGQA cp.async Q
        gather (not TMA), the SASS smem-dst applies an additional per-row XOR of
        ((row & 9) << 3) on the element index (measured exact against the
        runtime gather). Apply it so the WGMMA A-read lands on
        the cells the gather actually wrote."""
        i32 = self.ir.IntType(32)
        swiz_mask = (1 << swiz_B) - 1   # B3->7 (SW128), B2->3 (SW64), B1->1 (SW32)
        rowK = IRBuilder.mul(row_val, self.ir.Constant(i32, K_total), f"{nm}_rowK")
        e = IRBuilder.add(rowK, k_val, f"{nm}_e")
        # compact_swizzle: the cp.async-GATHERED narrow tiles (hd96 paged K/V) use the
        # COMPACT swizzle Swizzle<B,B+1,3> (yyy>>(B+4), xor<<(B+1)) instead of cute's
        # nominal Swizzle<B,4,3>, AND carry NO PackGQA (row&9)<<3 extra term (the
        # swizzle IS the whole layout). Decoded exact vs the runtime gather (K-read
        # read_hd==kc, V-read 20480/20480). For SW128 (B=3) the compact form is
        # IDENTICAL to the default (B+1==4, B+4==7), so SW128 stays byte-identical —
        # but callers gate it on the gathered narrow case regardless.
        _ysh = (swiz_B + 4) if compact_swizzle else 7
        _xsh = (swiz_B + 1) if compact_swizzle else 4
        yyy = IRBuilder.and_(IRBuilder.lshr(e, self.ir.Constant(i32, _ysh), f"{nm}_ys"),
                             self.ir.Constant(i32, swiz_mask), f"{nm}_yyy")
        xor = IRBuilder.shl(yyy, self.ir.Constant(i32, _xsh), f"{nm}_xor")
        es = IRBuilder.xor(e, xor, f"{nm}_es")
        if compact_swizzle:
            extra_row_swizzle = False   # gathered narrow: no PackGQA extra term
        if extra_row_swizzle and byte_per_elem != 1:
            # fp16/bf16 cp.async Q/K gather: the SASS smem-dst applies an extra
            # per-row XOR of ((row & 9) << 3) on the element index beyond cute's
            # Swizzle<3,4,3> (measured exact vs the runtime gather, K_total=64).
            # The fp8 (e4m3) gather is the 128-wide SW128 atom (K_total=128,
            # byte_per_elem=1) and writes the PLAIN cute Sw<3,4,3> layout with NO
            # extra row swizzle — re-probed against the runtime fp8 LDGSTS gather:
            # the element-
            # level discrepancy gather_elem ^ cute_elem is 0 for all 128 query rows
            # / 128 hdim. So gate the extra swizzle off for fp8; applying the fp16
            # ((row&9)<<3) to the fp8 A-read made odd / bit-3 query rows read the
            # wrong Q byte (corr 0.53). fp16/bf16 gather kernels (byte_per_elem=2)
            # keep the swizzle and stay byte-identical.
            rsw = IRBuilder.shl(IRBuilder.and_(row_val, self.ir.Constant(i32, 9), f"{nm}_rsm"),
                                self.ir.Constant(i32, 3), f"{nm}_rsw")
            es = IRBuilder.xor(es, rsw, f"{nm}_esr")
        if byte_per_elem == 1:
            eb = es                                                    # *1 fp8
        else:
            eb = IRBuilder.shl(es, self.ir.Constant(i32, 1), f"{nm}_eb")  # *2 fp16
        return IRBuilder.add(tile_base_byte, eb, f"{nm}_off")

    def _wgmma_sw128_product(self, desc_op, desc_hi_op, c_regs, defs,
                             n_dim, k_dim, IRBuilder, IRRegs, _get_val, a_regs=None,
                             is_bf16=False, is_fp8=False,
                             a_transposed=False, b_transposed=False,
                             swiz_B_a=3, swiz_B_b=3):
        """SW128 (128B-swizzle) descriptor-product decode — FlashAttention's real
        Q/K/V layout. See _try_wgmma_descriptor_product for the layout note.

        The descriptor low word's start_address (bits[13:0]) = (smem_byte>>4).
        cute steps it +2 per accumulating HGMMA (one per 16-k block). The tile
        base is 128B-aligned (start_address multiple of 8), so:
          k_block      = (start_address & 7) >> 1
          tile_base_by = (start_address & ~7) << 4
        SW128 fp16 has a fixed 64-element K-atom, so K_total = 64."""
        if desc_hi_op is None:
            return False
        i32 = self.ir.IntType(32)
        f32 = self.ir.FloatType()
        try:
            lo_val = _get_val(desc_op, "wgmma_descA")
            hi_val = _get_val(desc_hi_op, "wgmma_descB")
        except Exception:
            return False
        if lo_val is None or hi_val is None:
            return False
        if lo_val.type != i32:
            lo_val = self._insert_type_cast_if_needed(IRBuilder, lo_val, i32, "wgmma_descA_i32")
        if hi_val.type != i32:
            hi_val = self._insert_type_cast_if_needed(IRBuilder, hi_val, i32, "wgmma_descB_i32")

        # SW128 swizzle atom K-extent: fp16/bf16 = 64 elements (128B / 2B),
        # fp8 (e4m3) = 128 elements (128B / 1B). Verified against CUTE's
        # ss_smem_selector<K, Element, M, K>.
        # For fp16/bf16 the atom width is per-operand: K_total = 8<<swiz_B
        # (SW128 B3->64, SW64 B2->32, SW32 B1->16). swiz_B_a/b default 3 so SW128
        # kernels keep K_total=64 (byte-identical). fp8 keeps the fixed 128-wide
        # atom for both operands (its 1-byte e4m3 decode is a separate path).
        K_total = 128 if is_fp8 else 64           # legacy default (fp8 + fallback)
        K_total_a = 128 if is_fp8 else (8 << swiz_B_a)
        K_total_b = 128 if is_fp8 else (8 << swiz_B_b)
        # The descriptor low word steps +2 per accumulating MMA; each step is one
        # k-block = k_dim elements (16 for fp16, 32 for fp8). So k_global =
        # k_dim * k_block + k_local.
        kblk_stride = k_dim  # 16 (fp16) or 32 (fp8)

        def split(v, nm):
            sa = IRBuilder.and_(v, self.ir.Constant(i32, 0x3fff), f"{nm}_sa")
            kblk = IRBuilder.lshr(IRBuilder.and_(sa, self.ir.Constant(i32, 7), f"{nm}_kb0"),
                                  self.ir.Constant(i32, 1), f"{nm}_kblk")
            base = IRBuilder.shl(IRBuilder.and_(sa, self.ir.Constant(i32, ~0x7 & 0x3fff), f"{nm}_b0"),
                                 self.ir.Constant(i32, 4), f"{nm}_base")
            return base, kblk

        a_base, a_kblk = split(lo_val, "wg_a")
        # B descriptor is desc_hi_op (gdesc_UR + 2) for BOTH SS and RS forms —
        # cute lays the wgmma descriptor pair so the B low word is UR(n+2). VERIFIED
        # by the b_base trace: for the FA PV GEMM (RS) split(desc_hi_op=UR18) steps
        # through V's smem (2048,4096,...) while split(desc_op=UR16) is the stale Q
        # base (92160). (An earlier RS-only `b_base=desc_op` was wrong: it read Q as V
        # -> O=P·Q, spiky hdim 0-63 + zero hdim 64-127.)
        b_base, b_kblk = split(hi_val, "wg_b")   # B (K or V) descriptor = desc_hi_op
        # k_global = kblk_stride*k_block + k_local; both descriptors step together
        # (kblk_stride = 16 for fp16, 32 for fp8).
        a_kg0 = IRBuilder.mul(a_kblk, self.ir.Constant(i32, kblk_stride), "wg_akg0")
        b_kg0 = IRBuilder.mul(b_kblk, self.ir.Constant(i32, kblk_stride), "wg_bkg0")

        # Per-thread D-fragment (m,n) from cute CLayout_64xN (same as K-INTER).
        tid = self._load_const_mem_u32(IRBuilder, 0x2C, "wgmma_tid")  # SR_TID.X
        t0 = IRBuilder.and_(tid, self.ir.Constant(i32, 3), "wg_t0")
        t1 = IRBuilder.and_(IRBuilder.lshr(tid, self.ir.Constant(i32, 2), "wg_t1s"),
                            self.ir.Constant(i32, 7), "wg_t1")
        # warpgroup-LOCAL warp index (0..3). SR_TID.X is block-global, so a warp-
        # specialized kernel's consumer warpgroups have tid>=128; without &3 the warp
        # index is 4..11 and row m=t1+16*t2 lands >=64, outside the 64-row WGMMA tile
        # (FA3 consumers then read/store garbage). t0/t1 use bits[0:5] = already local.
        t2 = IRBuilder.and_(IRBuilder.lshr(tid, self.ir.Constant(i32, 5), "wg_t2s"),
                            self.ir.Constant(i32, 3), "wg_t2")
        m_warp = IRBuilder.add(t1, IRBuilder.mul(t2, self.ir.Constant(i32, 16), "wg_mw0"),
                               "wg_mwarp")
        n_lane = IRBuilder.mul(t0, self.ir.Constant(i32, 2), "wg_nlane")

        d_count = len(defs)
        n_oct = max(n_dim // 8, 1)
        coords_v = [((v // 2) % 2, v % 2, v // 4) for v in range(d_count)]
        if n_oct * 4 != d_count:
            return False

        # RS-form A (FA PV GEMM: A=P in registers, e.g. HGMMA D, R216, gdesc.tnspB, C).
        # x86 fibers don't share registers, so spill each thread's 4 f16x2 A regs to
        # per-warpgroup scratch (slifter_rs_a_spill), then read A[m,k] from scratch
        # (slifter_rs_a_read) instead of from a_base smem. SS-form (a_regs empty, e.g.
        # QK GEMM A=Q smem) keeps the smem read.
        i16 = self.ir.IntType(16)
        half = self.ir.HalfType()
        is_rs = bool(a_regs)

        # Rolling capture: `_fa_kblockn` = the N of the most recent non-bwd
        # SS-form GEMM lifted in this function. Despite the name this is NOT
        # always kBlockN — the value each is_mn_b consumer reads is order-
        # dependent, and the exec-verified corpus (63/63 FA3) depends on
        # exactly these values:
        #   * RS-form is_mn_b GEMMs (fwd PV, hd128/192/256) read the PREVIOUS
        #     SS GEMM's N = the QK GEMM's N = the true kBlockN (the QK decode
        #     precedes PV in the mainloop, so this is set by then).
        #   * SS-form is_mn_b GEMMs SELF-capture: this gate fires at the top of
        #     their own lift, so their else-branch atom stride uses their OWN
        #     n_dim (e.g. the hd64/128 bwd dQ B-read — those bwd classes have
        #     no `.tnspA`, so fa_bwd is False and this gate is live there).
        # A single per-function pre-scan "true kBlockN" was considered and
        # REJECTED: it would silently rewrite the self-capture consumers'
        # strides the corpus was exec-verified against (see
        # known limitation, L6).
        # Consumers for which this rolling value is provably wrong divert
        # explicitly instead: _fwd_vg_narrow / _narrow_bwd_rs_dkv ->
        # _fa_pds_atom, dq_kread -> _fa_kblockn_dqk, bwd SS-dKV ->
        # _fa_kblockn_bwd (below).
        # In the bwd-with-`.tnspA` classes the SS GEMMs' N is kBlockM
        # (=queries) under SwapAB, NOT kBlockN, so the capture is gated off —
        # the bwd RS dKV path keys off _fa_kblockm (per-function pre-scan).
        fa_bwd = bool(getattr(self, "_fa_bwd", False))
        if not is_rs and not fa_bwd:
            self._fa_kblockn = n_dim
        # In the bwd the SS-form dKV B-read (is_mn_b, dKV_swapAB=1) needs the real
        # kBlockN for MN_ATOM_STRIDE. The fwd `_fa_kblockn` capture above is gated
        # OFF for bwd (its SS GEMMs' N is kBlockM, not kBlockN), so without this the
        # stride would fall back to 176 and overshoot the dS tile (smem-guard crash).
        # `_fa_kblockn_bwd` is captured in the per-function pre-scan from the SS-form
        # dKV GEMM's N (the GEMM carrying both `.tnspA` and `.tnspB`); it is None for
        # the RS-form dKV class (hd64/128), so those stay byte-identical.
        # The 176 fallback fires only when NO non-bwd SS GEMM precedes this
        # instruction in the function (non-FA GEMM microbenches, e.g. the bench
        # wgmma/ws_attn suites) — every FA kernel lifts an SS GEMM (QK/SdP)
        # before any is_mn_b read. 176 = the FA fwd hd64/128 kBlockN the bench
        # microbench suite was verified against; it is a legacy default, not a
        # derived quantity.
        kblockn = getattr(self, "_fa_kblockn", None) or 176
        if fa_bwd:
            _kbn_bwd = getattr(self, "_fa_kblockn_bwd", None)
            if _kbn_bwd is not None:
                kblockn = _kbn_bwd
        kblockm = getattr(self, "_fa_kblockm", None) or n_dim

        # The bwd dKV RS-form GEMM (dV=Pᵀ·dO, dK=dSᵀ·Q, Mma_dKV_is_RS) carries
        # `.tnspB` like the fwd PV, but reads a K-major Q/dO smem tile *transposed*:
        # the contraction index is the query (kBlockM) and the atom/col index is the
        # hdim, with MN_ATOM_STRIDE = kBlockM*64. The fwd PV V is physically MN-major
        # in smem (TmaMajorV=MN, contraction over seqlen, stride kBlockN*64) and uses
        # the existing is_mn_b formula — gated OFF _fa_bwd so it stays byte-identical.
        bwd_dkv_b = bool(b_transposed) and fa_bwd and is_rs and not is_fp8

        # The bwd dQ GEMM (dQ = dS·K, dQ_swapAB) is the SS-form `.tnspA` HGMMA:
        # A = K read MN-major (`.tnspA`), B = dS read plain K-major from the sdS smem
        # tile the dS STSM.MT88 wrote (SmemLayoutPdS). That tile's swizzle is NOT the
        # generic SW128 sw3 (`e ^ (((e>>7)&7)<<4)`, which fits the q*K_total-row K/V/Q
        # tiles): SmemLayoutPdS = `Sw<3,4,3> o (kBlockM,(64,2)):(64,(1,5120))`, i.e.
        # element e = query*64 + key_local with the *Sw<3,4,3>* swizzle
        # `e ^ (((e>>6)&7)<<3)` (probe_pds_form/probe_pds_all vs cute, 0/10240). The
        # generic sw3 read missed it -> the dS B-read landed key^8 off on odd queries
        # (q-bit0 <-> key-bit3 coupling) while the store wrote the true cute layout, so
        # dQ came out uncorrelated. Gated _fa_bwd + SS `.tnspA` (uniquely the dQ GEMM;
        # the bwd QK/SdP SS GEMMs have no `.tnspA`, the dKV GEMMs are RS-form) so the
        # fwd and every other SW128 read stays byte-identical.
        dq_pds_b = (fa_bwd and bool(a_transposed) and not is_rs
                    and not bool(b_transposed) and not is_fp8)

        # The OTHER dQ GEMM layout: dQ_swapAB=FALSE (hdim64 non-causal, and the whole
        # softcap class). There dQ = dS·K with A major = PdS_Major(K), B major = MN, so
        # cute spells it `HGMMA D, gdesc.tnspB, C` — SS-form (A from smem), b_transposed
        # (B=K read MN-major), NOT a_transposed (no `.tnspA` at all in these kernels, so
        # `_fa_bwd` is False — we key on UBLKRED instead). Here the **A operand IS dS**,
        # read from the STSM.MT88-written sdS tile, which is the SmemLayoutPdS layout
        # `Sw<3,4,3> o (kBlockM,(kBlockN/(NumWG),NumWG))` = element e = query*64 +
        # key_local with swizzle `e ^ (((e>>6)&7)<<3)` and a per-key-WG atom jump
        # wgbase=(key>>6)*kBlockM*64 folded into the descriptor a_base. The generic
        # SW128 A-read (`sw>>7<<4`, K_total=64) mismatches it in 6144/8192 cells
        # (probe_pds_h64 vs cute, exact) — the q-bit0<->key-bit3 (key^8) coupling that
        # leaves the dQ output a clean PERMUTATION of the correct values (sorted-|mag|
        # relL2 0.002) but ~0.24 corr. Use the PdS swizzle for this A-read. Gated on
        # UBLKRED (bwd-only) + SS + b_transposed + !a_transposed + !fp8, which uniquely
        # matches this dQ GEMM (the bwd QK/SdP SS GEMMs have B=K plain K-major =>
        # b_transposed False; the dKV GEMMs are RS-form; fwd has no UBLKRED). dK/dV are
        # RS-spill GEMMs (unaffected), and the hdim128 dQ is `.tnspA` (a_transposed True)
        # => routes to dq_pds_b, not here. fwd byte-identical (no UBLKRED).
        has_ublkred = bool(getattr(self, "_has_ublkred", False))
        dq_pds_a = (has_ublkred and not is_rs and bool(b_transposed)
                    and not bool(a_transposed) and not is_fp8)

        # The bwd dQ GEMM (dQ_swapAB=0) B operand = K read from SmemLayoutKt
        # (composition(SmemLayoutK, transpose)). For hd64/128 the K atom K-extent =
        # kHeadDim/(NumWG/AtomLayoutMdQ) == the full hdim (64), so Kt is a single-
        # atom transpose and the generic is_mn_b `kc*ATOM + n_in` already matches it
        # (committed + byte-identical). hd96's K atom is SW32 (16-wide, kHeadDim 96 /
        # (2/1) = 48 -> 16-elem atom) with SIX hd-atoms, so Kt interleaves as
        # `e = (hd%16) + (hd//16)*2048 + key*16` (probe_dqk2_h96 EXACT) -> the
        # within-HGMMA read is `sw1(kc + key*ATOM)` (hd//16 atom jump folded into
        # b_base), which is the TRANSPOSE of is_mn_b's `kc*ATOM + n_in`. Gate the
        # transposed dQ K-read on the narrow-atom dQ GEMM (dq_pds_a companion +
        # swiz_B_b < 3) so hd64/128 (SW128 K, swiz_B_b==3) stay byte-identical.
        dq_kread = dq_pds_a and swiz_B_b < 3

        # MN-major B detection. In FlashAttention the PV GEMM's B operand (V) is
        # MN-major (TmaMajorV = GMMA::Major::MN), whereas the QK GEMM's B (K) is
        # K-major. The B smem (probed exact vs cute SmemLayoutVt) is the TRANSPOSE
        # of the K-major layout: V_off(n=hdim, k=seqlen) = (n/64)*MN_ATOM_STRIDE +
        # sw128_elem(k_local*64 + n%64), with the seqlen-block folded into b_base.
        # The MN-major B operand is EXACTLY the RS-form PV's V; the QK GEMM's B (K)
        # is SS-form and K-major. (The old `n_dim == 128` heuristic broke for kernels
        # whose QK N is also 128 — e.g. PagedKV kBlockN=128 — wrongly routing the QK
        # K-read through the MN-major path.) MN_ATOM_STRIDE = kBlockN*64 (the V tile's
        # seqlen extent * 64): 176*64 for kBlockN=176, 128*64 for kBlockN=128.
        # fp8 (e4m3): V is row-major in gmem and TRANSPOSED in-kernel (LDSM.T/STSM)
        # into a K-major Vt smem tile, so MmaMajorV == GMMA::Major::K (FA3
        # mainloop_fwd_sm90_tma_gmma_ws.hpp L65: `!Is_FP8 ? MN : K`). The PV GEMM's
        # B (Vt) is therefore K-major like the QK K-read — NOT the MN-major path
        # fp16/bf16 V uses. So is_mn_b stays False for fp8 even in RS-form.
        #
        # MN-major V is signalled by the descriptor's `.tnspB` suffix (GMMA::Major::MN
        # read of a K-major smem tile), NOT by RS-vs-SS form. The PV GEMM carries
        # `.tnspB` (MN-major V) and the QK GEMM does not (K-major K) regardless of
        # whether the MMA is RS- or SS-form. The old `is_rs` proxy held only because
        # every hdim128 PV happens to be RS-form-with-.tnspB; the hdim96 PV is the
        # first SS-form-PV-WITH-.tnspB (HGMMA.64x96x16 gdesc[UR].tnspB, A=P from smem
        # => is_rs=False), so `is_rs` wrongly read its V K-major. Keying on the suffix
        # (b_transposed) is byte-identical for every hdim128 kernel (RS PV: is_rs==
        # b_transposed==True) and QK (both False), and fixes the SS-form PV. fp8 keeps
        # is_mn_b False (Vt is K-major in smem; `and not is_fp8`).
        #
        # The SS-form dKV GEMM (dKV_swapAB=1: dK=dSᵀ·Q, dV=Pᵀ·dO, e.g. hdim192's
        # HGMMA.64x96x16.tnspA.tnspB) reads BOTH operands from smem. A=Q/dO read
        # MN-major (`.tnspA`, the existing a_transposed path), B=dS/P read with
        # PdSt_Major=MN — i.e. the SmemLayoutPdSt = composition(SmemLayoutPdS,
        # transpose) layout, NOT the V-style MN-major SW128 model. SmemLayoutPdS
        # here is `Sw<2,4,3> o ((8,8),(32,3)):((32,256),(1,2048))` over (query=64,
        # key=96) — an SW64 (atom width 32) swizzle, not SW128 (atom 64). Routing
        # this B-read through is_mn_b (SW128, atom=64) reads the wrong dS cells and
        # leaves dK/dV numerically wrong. Use the PdSt formula instead (probed EXACT
        # vs cute SmemLayoutPdSt, probe_pdst_h192, 6144/6144). Gated on the SS-form
        # dKV GEMM uniquely = `.tnspA` AND `.tnspB` AND SS-form (the dQ GEMM is
        # `.tnspB`-only; QK/SdP have no tnsp; the RS-form dKV class — hd64/128 — is
        # is_rs). fwd has no `.tnspA` (fa_bwd False) so it stays byte-identical.
        dkv_pdst_b = (fa_bwd and bool(a_transposed) and bool(b_transposed)
                      and not is_rs and not is_fp8)
        is_mn_b = b_transposed and not is_fp8 and not dkv_pdst_b
        # (The per-GEMM MN atom stride — conceptually kblockn*ATOM_b — is
        # computed per branch at the atom_off site below; comments referring to
        # "MN_ATOM_STRIDE" mean that quantity.)
        # fwd PV V-read of a cp.async-GATHERED narrow-swizzle (SW64/SW32) V tile
        # (known instance: hd96 paged). Uniquely:
        # is_mn_b (`.tnspB` V) AND SS-form (not is_rs — hd96's kBlockM=192 spills P
        # to smem so its PV is SS, unlike hd128/192/256's RS-form PV) AND fwd AND
        # narrow swizzle (swiz_B_b<3 = SW64/SW32, only hd96) AND V cp.async-gathered
        # (paged). The cp.async gather writes V with a layout DIFFERENT from cute's
        # nominal TMA SmemLayoutV — decoded noise-free against 20,480 runtime
        # gather cells using V-smem source-id injection:
        #     G_elem(key_local, hd) = (hd>>5)*(kBlockN*ATOM) + sw(key_local*ATOM + hd&31)
        # with ATOM = 8<<swiz_B_b (32 for SW64) and the COMPACT gather swizzle
        #     sw(e) = e ^ (((e >> (B+4)) & ((1<<B)-1)) << (B+1))      # Swizzle<B,B+1,3>
        # (B=swiz_B_b). This differs from is_mn_b's default Swizzle<B,4,3> (yyy>>7,
        # xor<<4) in the MBase (B+1 not fixed 4) — the TMA-loaded SW64 bwd dO/Q keeps
        # Swizzle<B,4,3>, so the compact form is GATHER-specific — AND drops the
        # PackGQA (row&9)<<3 extra swizzle (the gather has none; that term corrupts
        # the V hd read, making read_hd kc-dependent). The hd-atom stride uses the
        # TRUE kBlockN (=_fa_pds_atom, the QK N=128), NOT the SS-PV's own N=96 which
        # clobbers `_fa_kblockn` (SS-form PV re-enters the not-is_rs `_fa_kblockn`
        # capture). Gated so SW128 fwd (swiz_B_b==3), TMA-V, RS-form PV (hd128/192/
        # 256), fp8, and all bwd stay byte-identical.
        _fwd_vg_narrow = (is_mn_b and not is_rs and not fa_bwd and not is_fp8
                          and swiz_B_b < 3
                          and bool(getattr(self, "_v_cpasync_gathered", False)))
        # fwd QK K-read of a cp.async-GATHERED narrow-swizzle K tile (SS-form,
        # B=K read K-major; known instance: hd96 paged, SW64).
        # The K tile is gathered with the SAME compact Swizzle<B,B+1,3> layout as V
        # (decoded: read_hd == kc for all keys only under the compact swizzle; the
        # default Swizzle<B,4,3>+ersb gives a key-dependent hd shift = a per-key QK
        # dot-product error, softmax-amplified to O corr ~0.90). Same gate as V but
        # for the NON-transposed B-read (QK); the PV V-read is is_mn_b (b_transposed).
        _fwd_kg_narrow = (not is_rs and not fa_bwd and not is_fp8 and not b_transposed
                          and swiz_B_b < 3
                          and bool(getattr(self, "_v_cpasync_gathered", False)))
        # fwd QK A=Q read of a cp.async-GATHERED narrow-swizzle Q tile (known
        # instance: hd96 paged): Q is cp.async-GATHERED into smem with the SAME
        # compact Swizzle<B,B+1,3> layout as K/V (decoded: with the default Sw<2,4,3>
        # +ers read the Q-read has a query-dependent hd shift read_hd=kc^(8*qbit1); the
        # compact swizzle makes read_hd==kc). Use the compact swizzle (which also drops
        # the spurious ers), matching the K-read. Gated like _fwd_kg_narrow but on the
        # A=Q side (not is_mn_b = the QK GEMM, whose B=K is not transposed; the PV A=P
        # read is is_mn_b so excluded, keeping its σ correction).
        _fwd_qg_narrow = (not is_rs and not fa_bwd and not is_fp8 and not is_mn_b
                          and not b_transposed and swiz_B_a < 3
                          and bool(getattr(self, "_v_cpasync_gathered", False)))
        # bwd dP GEMM A=V read swizzle (narrow SW64 B-tile; known instance:
        # hd96). The dP GEMM (dP = dOᵀ·V,
        # SdP_swapAB) is the UNIQUE RS-form HGMMA with NO transpose suffix across the
        # whole bwd family: hd64 dP is SS-form; hd128 RS is `.tnspB`; hd192 RS is
        # `.tnspA.tnspB`; hd256 RS is `.tnspA`/`.tnspA.tnspB`. Only hd96 spells dP as
        # `HGMMA R120, R204, gdesc[UR12]` (RS, A=V regs from LDSM, B=dO desc, no tnsp).
        # Its A operand (V) is read by `LDSM.16.M88.4` from a SW64 (Sw<2,4,3>, ATOM=32)
        # smem tile; the lifter's LDSM redistribution + rs_a_spill stores the V
        # fragment at SW64-SCRAMBLED scratch slots: physical scratch[m,k] holds
        #   V[key = m ^ ((m>>3)&1),  hd_in = k ^ (((m>>1)&1) << 3)]
        # (the Swizzle<2,4,3> within-atom key↔hd coupling, the SAME family as the
        # committed dV-store slifter_dkv_stsm_sw64 + the PdS bit1/bit3 coupling).
        # Decoded noise-free via V-smem source-id injection
        # (SLIFTER_VID_INJECT) joined with the RS-A spill, 2048/2048.
        # So to read logical V[key=m_val, hd_in=kc] we must read the SCRAMBLED slot
        #   m = m_val ^ ((m_val>>3)&1);  k = kc ^ (((m_val>>1)&1) << 3)
        # (a self-consistent inverse — m bit3 / m bit1 are unaffected by the flips).
        # Gated: RS-form, NO transpose, bwd (has_ublkred) — unique to hd96 dP — AND the
        # B tile is SW64 (swiz_B_b == 2). The dP GEMM's gdesc[UR12] is the B operand
        # (dO); its high word UR15 = 0x80000020 (lt2 = SW64 B2). V (the RS A operand,
        # loaded by LDSM) shares hd96's SW64 layout (driver sets dO+V swiz_b=2). The
        # A.hi UR13 = 0x40000040 (a reused dKV-store word, lt1 = SW128) so swiz_B_a is
        # NOT a reliable V signal here — key on swiz_B_b. Any SW128 dP stays B=3 ->
        # byte-identical.
        dp_av_sw64 = (is_rs and not a_transposed and not b_transposed
                      and bool(getattr(self, "_has_ublkred", False))
                      and not is_fp8 and swiz_B_b == 2)
        if is_rs:
            def _reg_i32(op, nm):
                v = _get_val(op, nm)
                t = v.type
                if t == i32:
                    return v
                if isinstance(t, self.ir.PointerType):
                    return IRBuilder.ptrtoint(v, i32, nm + "_p")
                w = getattr(t, "width", None)
                if w is not None:
                    if w == 32:
                        return IRBuilder.bitcast(v, i32, nm + "_bc")
                    if w < 32:
                        return IRBuilder.zext(v, i32, nm + "_z")
                    return IRBuilder.trunc(v, i32, nm + "_t")
                return IRBuilder.bitcast(v, i32, nm + "_vbc")  # v2f16 / float (32-bit)
            rs_vals = []
            for r in range(4):
                if r < len(a_regs):
                    rs_vals.append(_reg_i32(a_regs[r], f"wg_arg{r}"))
                else:
                    rs_vals.append(self.ir.Constant(i32, 0))
            # fp8 (e4m3) QGMMA: the 4 A regs each pack 4 e4m3 bytes (K-step 32)
            # with a different A-fragment layout than fp16's f16x2 packing, so
            # spill via the dedicated fp8 path. fp16/bf16 RS-A is unchanged.
            _spill_fn = "slifter_rs_a_spill_fp8" if is_fp8 else "slifter_rs_a_spill"
            self.run_intrinsic(_spill_fn, IRBuilder, rs_vals, "rs_a_spill")

        # DEBUG (gated SLIFTER_TMA_DEBUG at lift time, same env the wrapper's
        # slifter_wgmma_dbg checks at runtime): trace the resolved A/B descriptor
        # bases per GEMM. Production lifts carry no call.
        if os.environ.get("SLIFTER_TMA_DEBUG"):
            _dbg_hi, _ = split(hi_val, "wg_dbg_hi")  # split(desc_hi_op=UR+2) for the trace
            self.run_intrinsic("slifter_wgmma_dbg", IRBuilder,
                               [a_base, b_base, _dbg_hi,
                                self.ir.Constant(i32, 1 if is_rs else 0)], "wgmma_dbg")

        for i, d in enumerate(defs):
            v1, v0, v2 = coords_v[i]
            m_val = IRBuilder.add(m_warp, self.ir.Constant(i32, 8 * v1), f"wg_m{i}")
            n_val = IRBuilder.add(n_lane, self.ir.Constant(i32, v0 + 8 * v2), f"wg_n{i}")
            acc = self.ir.Constant(f32, 0.0)
            if i < len(c_regs):
                acc = _get_val(c_regs[i], f"wgmma_c{i}")
                if acc.type != f32:
                    acc = IRBuilder.bitcast(acc, f32, f"wgmma_c{i}_f32")
            if is_mn_b or bwd_dkv_b:
                # The MN-major tile (fwd V; bwd dKV dO/Q) splits the n (hdim) axis
                # into atoms of width ATOM_b = 8<<swiz_B_b (SW128 B3->64, SW64
                # B2->32, SW32 B1->16). swiz_B_b defaults 3 so SW128 kernels keep
                # the 64-wide atom (n_atom = n>>6, n_in = n&63, stride *64) BYTE-
                # IDENTICAL. hd96's dO/Q is SW64 (probe_dkvb_h96: B=2 ATOM=32 atom
                # stride kBlockM*32 = kBlockM*ATOM_b).
                _ATOM_b = 128 if is_fp8 else (8 << swiz_B_b)
                _atom_log2 = _ATOM_b.bit_length() - 1     # 6 for 64, 5 for 32, 4 for 16
                n_atom = IRBuilder.lshr(n_val, self.ir.Constant(i32, _atom_log2), f"wg_natom{i}")
                n_in = IRBuilder.and_(n_val, self.ir.Constant(i32, _ATOM_b - 1), f"wg_nin{i}")
                # fwd PV V: atom stride = kBlockN*ATOM (V's seqlen extent). The SS
                # `.tnspA.tnspB` dKV (hd192/256) uses bwd_dkv_b (kBlockM*ATOM). The
                # RS-form dKV (hd64/96/128, dKV_swapAB=0, no `.tnspA`) reads dO/Q via
                # is_mn_b: for hd64/128 the dO tile is SW128 (ATOM=64) and the
                # existing kblockn*ATOM stride is correct (committed + byte-identical).
                # hd96's dO is SW64 (ATOM=32, the mixed-swizzle case) and its tile is
                # [kBlockM=64, hdim] (kBlockM != kBlockN=128), so its hdim atoms are
                # kBlockM*ATOM = 64*32 = 2048 apart, NOT kblockn*32=4096
                # (probe_mnb_h96 EXACT). Divert the stride to kBlockM only for the
                # NARROW-atom (swiz_B_b < 3) bwd RS dKV read so hd64/128 (SW128,
                # swiz_B_b==3) stay byte-identical. kBlockM == the plain SdP GEMM's N
                # == `_fa_pds_atom` (64 for hd96).
                _narrow_bwd_rs_dkv = (bool(getattr(self, "_has_ublkred", False))
                                      and is_rs and bool(b_transposed) and not is_fp8
                                      and swiz_B_b < 3)
                if bwd_dkv_b:
                    _atom_stride = kblockm * _ATOM_b
                elif _narrow_bwd_rs_dkv:
                    _kbm = getattr(self, "_fa_pds_atom", None) or kblockn
                    _atom_stride = _kbm * _ATOM_b
                elif _fwd_vg_narrow:
                    # TRUE kBlockN = the QK GEMM's N (=_fa_pds_atom, 128); the SS-form
                    # PV clobbers `_fa_kblockn` to its own N=96, halving the hd-atom
                    # stride and bleeding the hd-atom jump into the key dimension.
                    _kbn_v = getattr(self, "_fa_pds_atom", None) or kblockn
                    _atom_stride = _kbn_v * _ATOM_b
                else:
                    _atom_stride = kblockn * _ATOM_b
                atom_off = IRBuilder.mul(n_atom, self.ir.Constant(i32, _atom_stride), f"wg_aoff{i}")
            for k in range(k_dim):
                kc = self.ir.Constant(i32, k)
                if not is_rs:
                    kg_a = IRBuilder.add(a_kg0, kc, f"wg_kga{i}_{k}")
                    # PackGQA cp.async-gather-loaded Q smem carries an extra
                    # ((row & 9) << 3) element swizzle vs the TMA SW128 layout;
                    # mirror it in the A-read so the QK GEMM reads the right query
                    # row (else odd / bit-3 M-rows read query+8). Gated on the
                    # kernel actually using LDGSTS — base TMA kernels are unaffected.
                    _ers = getattr(self, "_q_gather_loaded", False)
                    if a_transposed:
                        # SS `.tnspA`: the A operand is read MN-major (cute
                        # GMMA::Major::MN) from a K-major SW128 tile. Probed exact
                        # vs cute (probe_ss_tnspa.cpp): the MN read is the SAME
                        # physical SW128 layout with the (row, k) coordinate SWAPPED
                        # — pass the k index as "row" and the m index as "k".
                        # (The SwapAB SdP/dQ GEMMs: phys-A = K read MN-major.)
                        row_a = kg_a
                        # hd256 dQ swapAB GEMM (dq_pds_b, atom-40 PdS, kBlockN=80):
                        # the A=K `.tnspA` row IS the contraction key. The dS tile is
                        # STSM-written by the two SdP warpgroups; WG1 (keys >= kBlockN/2
                        # = 40) stores each (even,odd) key PAIR in SWAPPED order, and the
                        # dQ GEMM contracts in that dS-storage key order. The K tile is
                        # TMA-loaded in PLAIN key order, so for the WG1 half the A-read
                        # lands on key^1 while the dS B-read (un-swapped by its own
                        # formula) supplies the LOGICAL key -> A and B contract MISALIGNED
                        # keys for keys 40..79 -> dQ corr 0.527 (dK/dV unaffected: they use
                        # the RS-spill dS / dkv_pdst_b path). PROVEN noise-free (K-smem
                        # source-id injection + the SOP a_off/b_off dump): mismatch
                        # <=> Bkey>=40 EXACTLY (100%), and Akey == Bkey^1 for all of
                        # them. Fix: mirror the WG1
                        # key^1 un-swap on the K A-read for the absolute key >= 40.
                        # The absolute contraction key is NOT in kg_a (= the LOCAL key
                        # 0..15; the descriptor low-word block bits are 0 here) — it is in
                        # a_base, which the source-id decode proved is EXACTLY
                        # block*128 bytes (a_base 0/2048/4096/6144/8192 = key-block
                        # 0/16/32/48/64). So abs_key = (a_base >> 7) + kg_a, and since the
                        # key-block is a multiple of 16 (even), XOR-ing kg_a's low bit
                        # flips abs_key's low bit. Gated on dq_pds_b (uniquely the dQ
                        # swapAB GEMM: a_transposed AND NOT b_transposed; the dKV GEMM is
                        # `.tnspA.tnspB` => dq_pds_b False) AND the atom-40 PdS (hd256:
                        # 32 < _fa_pds_atom <= 48) so hd64/96/128/192 (and the dKV/QK/SdP
                        # reads) stay BYTE-IDENTICAL. khalf = kBlockN/2.
                        _pa = getattr(self, "_fa_pds_atom", None) or 64
                        if dq_pds_b and 32 < _pa <= 48:
                            # abs_key = key-block + kg_a. The key-block is in a_base as
                            # block_key*128 bytes (a_base>>7 = block_key), BUT a_base ALSO
                            # carries the hd-ATOM jump (atom*rows*64*2 bytes = atom*kBlockN
                            # in a_base>>7 units). Reduce that out with `% kBlockN` so the
                            # WG1 test sees only the key-block — else hd-atoms 1..3 read
                            # a_base>>7 >= kBlockN, the test always fires, and WG0 keys get
                            # wrongly swapped (proven: hd-atom 0 corr 1.0 but atoms 1..3
                            # ~0.46 before this reduce). kBlockN = _fa_kblockn_bwd = 80.
                            _khalf = self.ir.Constant(i32, kblockn >> 1)
                            _kblk_raw = IRBuilder.lshr(a_base, self.ir.Constant(i32, 7), f"wg_kablk{i}_{k}")
                            _kblk_abs = IRBuilder.urem(_kblk_raw, self.ir.Constant(i32, kblockn), f"wg_kabm{i}_{k}")
                            _abskey = IRBuilder.add(_kblk_abs, kg_a, f"wg_kabs{i}_{k}")
                            _is_wg1a = IRBuilder.icmp_unsigned(">=", _abskey, _khalf, f"wg_kaw1{i}_{k}")
                            _kxa = IRBuilder.select(_is_wg1a, self.ir.Constant(i32, 1),
                                                    self.ir.Constant(i32, 0), f"wg_kaxb{i}_{k}")
                            row_a = IRBuilder.xor(kg_a, _kxa, f"wg_kga_sw{i}_{k}")
                        a_off = self._sw128_phys_off(IRBuilder, a_base, row_a, m_val, K_total_a, f"sa{i}_{k}",
                                                     extra_row_swizzle=_ers,
                                                     byte_per_elem=(1 if is_fp8 else 2),
                                                     swiz_B=swiz_B_a)
                    elif dq_pds_a:
                        # bwd dQ (dQ_swapAB=0): A = dS read plain from the STSM.MT88-
                        # written sdS tile (SmemLayoutPdS), NOT a generic SW128 tile.
                        # The PdS swizzle depends on the atom K-extent =
                        # kBlockN/(NumWG/AtomLayoutMSdP) = `_fa_pds_atom` (the plain SS
                        # GEMM's N), probed EXACT vs cute:
                        #   atom 64 (hd64/softcap): Sw<3,4,3> over a 64-wide row,
                        #       e = query*64 + key_local; swizzle `e ^ (((e>>6)&7)<<3)`.
                        #       (probe_pds_h64, 0 mismatch.)
                        #   atom 32 (hd192): Sw<2,4,3> over a 32-wide atom,
                        #       e = (query%8)*32 + (query//8)*256 + (key%32)
                        #           + (key//32)*2048; swizzle `e ^ (((e>>7)&3)<<4)`.
                        #       (probe_pds_h192 PdS(query,key), 6144/6144.)
                        # The per-key-WG / key-block atom jump is folded into a_base by
                        # the descriptor start_address; kg_a is the in-GEMM key.
                        _pds_atom = getattr(self, "_fa_pds_atom", None) or 64
                        if _pds_atom <= 32:
                            # 32-wide PdS atom (hd192): the dS tile is STSM.16.M88.4
                            # (trans==0)-written, with the SAME true physical layout the
                            # dkv_pdst_b B-read targets (decoded noise-free, 6144/6144):
                            #   key2 = key ^ (8 if (query & 2) else 0)  # q-bit1 <-> k-bit3
                            #   e    = (query%8)*32 + (query//8)*256 + (key2%32) + (key2//32)*2048
                            #   phys = e ^ (((e>>7)&1)<<4)              # Sw<1,4,3>
                            # query = m_val, key = kg_a (both runtime i32). The coupling
                            # term (m_val&2)?8:0 = (m_val & 2) << 2.
                            kcouple = IRBuilder.shl(
                                IRBuilder.and_(m_val, self.ir.Constant(i32, 2), f"wg_pdsa_kc0{i}_{k}"),
                                self.ir.Constant(i32, 2), f"wg_pdsa_kc{i}_{k}")
                            key2 = IRBuilder.xor(kg_a, kcouple, f"wg_pdsa_k2{i}_{k}")
                            kin = IRBuilder.and_(key2, self.ir.Constant(i32, 31), f"wg_pdsa_ki{i}_{k}")
                            katom = IRBuilder.lshr(key2, self.ir.Constant(i32, 5), f"wg_pdsa_ka{i}_{k}")
                            qlo = IRBuilder.and_(m_val, self.ir.Constant(i32, 7), f"wg_pdsa_ql{i}_{k}")
                            qhi = IRBuilder.lshr(m_val, self.ir.Constant(i32, 3), f"wg_pdsa_qh{i}_{k}")
                            e_a = IRBuilder.add(
                                IRBuilder.add(
                                    IRBuilder.add(
                                        IRBuilder.mul(qlo, self.ir.Constant(i32, 32), f"wg_pdsa_qlo{i}_{k}"),
                                        IRBuilder.mul(qhi, self.ir.Constant(i32, 256), f"wg_pdsa_qhi{i}_{k}"),
                                        f"wg_pdsa_q{i}_{k}"),
                                    kin, f"wg_pdsa_e0{i}_{k}"),
                                IRBuilder.mul(katom, self.ir.Constant(i32, 2048), f"wg_pdsa_kam{i}_{k}"),
                                f"wg_pdsa_e{i}_{k}")
                            # Physical STSM swizzle is Sw<1,4,3> (bit7->bit4 only),
                            # not cute's nominal Sw<2,4,3> (decoded from the runtime
                            # dump; the bit8->bit5 XOR is absent in the physical write).
                            yya = IRBuilder.and_(
                                IRBuilder.lshr(e_a, self.ir.Constant(i32, 7), f"wg_pdsa_ys{i}_{k}"),
                                self.ir.Constant(i32, 1), f"wg_pdsa_yyy{i}_{k}")
                            esa = IRBuilder.xor(
                                e_a, IRBuilder.shl(yya, self.ir.Constant(i32, 4), f"wg_pdsa_xor{i}_{k}"),
                                f"wg_pdsa_es{i}_{k}")
                        else:
                            # Sw<3,4,3>, 64-wide row (hd64/softcap class).
                            e_a = IRBuilder.add(
                                IRBuilder.mul(m_val, self.ir.Constant(i32, 64), f"wg_pdsa_m64_{i}_{k}"),
                                kg_a, f"wg_pdsa_e{i}_{k}")
                            yya = IRBuilder.and_(
                                IRBuilder.lshr(e_a, self.ir.Constant(i32, 6), f"wg_pdsa_ys{i}_{k}"),
                                self.ir.Constant(i32, 7), f"wg_pdsa_yyy{i}_{k}")
                            esa = IRBuilder.xor(
                                e_a, IRBuilder.shl(yya, self.ir.Constant(i32, 3), f"wg_pdsa_xor{i}_{k}"),
                                f"wg_pdsa_es{i}_{k}")
                        eba = IRBuilder.shl(esa, self.ir.Constant(i32, 1), f"wg_pdsa_eb{i}_{k}")  # *2 fp16
                        a_off = IRBuilder.add(a_base, eba, f"sa{i}_{k}_pdsoff")
                    else:
                        _kg_a_read = kg_a
                        if _fwd_vg_narrow:
                            # hd96 fwd SS-form PV A=P read: the QK score C-fragment is
                            # spilled to smem in a layout whose contraction-key index
                            # has its low 2 bits reversed within each odd 8-block (bit3
                            # set) vs the cute SmemLayout the SW128 read assumes. Undo:
                            #   key' = key ^ (((key>>3)&1) * 3)
                            # Decoded exactly using a single dominant key and a
                            # softmax-onehot scan (0/256 mismatches). Only this GEMM's
                            # P-read (gated _fwd_vg_narrow)
                            # is affected — every other SS A-read is byte-identical.
                            _b3 = IRBuilder.and_(IRBuilder.lshr(kg_a, self.ir.Constant(i32, 3), f"wg_pk_b3s{i}_{k}"),
                                                 self.ir.Constant(i32, 1), f"wg_pk_b3{i}_{k}")
                            _sw = IRBuilder.mul(_b3, self.ir.Constant(i32, 3), f"wg_pk_sw{i}_{k}")
                            _kg_a_read = IRBuilder.xor(kg_a, _sw, f"wg_pk_kg{i}_{k}")
                        a_off = self._sw128_phys_off(IRBuilder, a_base, m_val, _kg_a_read, K_total_a, f"sa{i}_{k}",
                                                     extra_row_swizzle=_ers,
                                                     byte_per_elem=(1 if is_fp8 else 2),
                                                     swiz_B=swiz_B_a,
                                                     compact_swizzle=_fwd_qg_narrow)
                # The B operand (QK's K, PV's V) carries the same ((row & 9) << 3)
                # cp.async smem swizzle as the A-read when K/V are filled by cp.async
                # (PagedKVNonTMA) rather than TMA. Gated on _kv_gather_loaded so TMA
                # K/V kernels (PackGQA-only, split) are unaffected. For the QK K-read
                # the gather row is the key index (n_val); for the MN-major V-read the
                # gather row is the seqlen = k_local (kc).
                _ersb = getattr(self, "_kv_gather_loaded", False)
                if dkv_pdst_b:
                    # SS-form dKV B-read (dKV_swapAB=1): B = dS/P read from the
                    # STSM.16.M88.4-written sdS/sP smem tile. CRITICAL: for the 32-wide
                    # PdS atom (hd192) the dS/P tile is written by STSM.16.M88.4
                    # (trans==0, NON-transposed), NOT MT88 like hd128's 64-wide atom.
                    # The M88 gather places the SdP C-fragment so the TRUE physical
                    # layout (decoded noise-free from SLIFTER_STSM_FRAG: source
                    # lane/register -> smem offset, then matched to P, 6144/6144)
                    # is NOT cute's nominal SmemLayoutPdS Sw<2,4,3>. It is:
                    #   key2  = key ^ (8 if (query & 2) else 0)   # query-bit1 <-> key-bit3
                    #   e     = (query%8)*32 + (query//8)*256 + (key2%32) + (key2//32)*2048
                    #   phys  = e ^ (((e>>7)&1)<<4)               # Sw<1,4,3>
                    # The query-bit1 / key-bit3 coupling is the 32-wide M88 write
                    # signature: queries with bit1 set land their key columns XOR'd by 8.
                    # The 64-wide atom (hd64/128) uses MT88 and matched cute, so this
                    # path is hd192/hd256-specific (gated dkv_pdst_b). The contraction K
                    # is the within-MMA query = kc (k, 0..15); b_base advances +512
                    # elements per accumulating HGMMA (the query-block jump). qin is a
                    # Python int so the (qin & 2) coupling is folded at lift time.
                    qin = k                                    # within-MMA query 0..15 (Python int)
                    _pdst_atom = getattr(self, "_fa_pds_atom", None) or 32
                    if _pdst_atom <= 32:
                        # 32-wide PdS atom (hd192): bit1<->bit3 coupling + Sw<1,4,3>.
                        # key bit3 (value 8) flips when the query has bit1 set.
                        nin0 = IRBuilder.and_(n_val, self.ir.Constant(i32, 31), f"wg_pdstn{i}_{k}")
                        nin = (IRBuilder.xor(nin0, self.ir.Constant(i32, 8), f"wg_pdstnx{i}_{k}")
                               if (qin & 2) else nin0)
                        natom = IRBuilder.lshr(n_val, self.ir.Constant(i32, 5), f"wg_pdsta{i}_{k}")
                        e_b = IRBuilder.add(
                            IRBuilder.add(
                                self.ir.Constant(i32, (qin % 8) * 32 + (qin // 8) * 256),
                                nin, f"wg_pdste0_{i}_{k}"),
                            IRBuilder.mul(natom, self.ir.Constant(i32, 2048), f"wg_pdstam{i}_{k}"),
                            f"wg_pdste{i}_{k}")
                        yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 7), f"wg_pdsts{i}_{k}"),
                                             self.ir.Constant(i32, 1), f"wg_pdsty{i}_{k}")
                        es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, 4), f"wg_pdstx{i}_{k}"),
                                           f"wg_pdstes{i}_{k}")
                    else:
                        # 40-wide PdS atom (hd256, kBlockN/(NumWG/MSdP)=80/2=40). The
                        # dS/P tile is written by STSM.16.M88.2 (trans==0). Its TRUE
                        # physical layout (decoded noise-free via SLIFTER_STSM_FRAG;
                        # source lane/register per smem offset matched to unscaled-dS
                        # and P values, WG0 255/256) is PLAIN LINEAR
                        # with NO swizzle (the SW1/SW2 bit7->bit4 XOR is ABSENT and there
                        # is NO bit1<->bit3 coupling — verified for all 64 queries incl.
                        # q>=16 where Sw<1> would diverge):
                        #   e = (key>>3)*512 + query*8 + (key&7)
                        # i.e. key decomposed as (key&7, key>>3):(1,512), query stride 8.
                        # query = within-MMA query qin (b_base carries the query-block);
                        # key = n_val (global key 0..79, the *512 term lands WG1's keys
                        # 40-79 at +2560 = its sub-tile base, matching the dump).
                        # SECOND-WARPGROUP key-pair swap: the dS/P tile is filled by the
                        # two SdP warpgroups (WG0 keys 0..kBlockN/2-1, WG1 the rest); the
                        # WG1 sub-tile stores each (even,odd) key PAIR in SWAPPED order
                        # (decoded EXACT vs the SLIFTER_STSM_FRAG dump: WG1 e=2560 holds
                        # key 41 not 40, AND the lifted dV/dK output for keys 40..79 came
                        # back as a clean key^1 permutation, corr 1.0000). So XOR the key
                        # low bit for keys >= kBlockN/2. WG0 (keys 0..39) is byte-exact
                        # without it (verified perfect). kblockn = _fa_kblockn_bwd = 80.
                        khalf = self.ir.Constant(i32, kblockn >> 1)
                        _is_wg1 = IRBuilder.icmp_unsigned(">=", n_val, khalf, f"wg_pdstw1{i}_{k}")
                        _xb = IRBuilder.select(_is_wg1, self.ir.Constant(i32, 1),
                                               self.ir.Constant(i32, 0), f"wg_pdstxb{i}_{k}")
                        nin = IRBuilder.xor(
                            IRBuilder.and_(n_val, self.ir.Constant(i32, 7), f"wg_pdstn0{i}_{k}"),
                            _xb, f"wg_pdstn{i}_{k}")
                        kblk = IRBuilder.lshr(n_val, self.ir.Constant(i32, 3), f"wg_pdstkb{i}_{k}")
                        es = IRBuilder.add(
                            IRBuilder.add(
                                self.ir.Constant(i32, qin * 8),
                                nin, f"wg_pdste0_{i}_{k}"),
                            IRBuilder.mul(kblk, self.ir.Constant(i32, 512), f"wg_pdstkm{i}_{k}"),
                            f"wg_pdstes{i}_{k}")
                    eb2 = IRBuilder.shl(es, self.ir.Constant(i32, 1), f"wg_pdsteb{i}_{k}")  # *2 fp16
                    b_off = IRBuilder.add(b_base, eb2, f"sb{i}_{k}_off")
                elif bwd_dkv_b:
                    # bwd dKV `.tnspB`: B = dO/Q, a K-major SW128 tile read TRANSPOSED.
                    # Probed exact vs cute (probe_rs_tnspb.cpp, 0/36): the contraction
                    # index is the query q = b_kg0 + kc (b_kg0 = b_kblk*16 carries the
                    # query-block advance the descriptor start_address encodes), the
                    # atom/col index is the hdim, and the within-atom element is
                    # sw3(q*64 + n_in) with atom stride kBlockM*64. (No cp.async
                    # gather swizzle: bwd dO/Q are TMA-loaded.)
                    # within-atom element = sw(q*ATOM_b + n_in); ATOM_b = 8<<swiz_B_b
                    # (SW128 64, SW64 32). swiz_B_b default 3 keeps the 64-wide form
                    # byte-identical for hd128/192/256; hd96 dO/Q is SW64 (ATOM 32).
                    _ATOM_b = 128 if is_fp8 else (8 << swiz_B_b)
                    _bmask = (1 << swiz_B_b) - 1
                    q_idx = IRBuilder.add(b_kg0, kc, f"wg_q{i}_{k}")
                    e_b = IRBuilder.add(IRBuilder.mul(q_idx, self.ir.Constant(i32, _ATOM_b), f"wg_q64_{i}_{k}"),
                                        n_in, f"wg_qeb{i}_{k}")
                    yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 7), f"wg_qys{i}_{k}"),
                                         self.ir.Constant(i32, _bmask), f"wg_qyyy{i}_{k}")
                    es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, 4), f"wg_qxor{i}_{k}"),
                                       f"wg_qes{i}_{k}")
                    elem = IRBuilder.add(atom_off, es, f"wg_qbelem{i}_{k}")
                    eb2 = IRBuilder.shl(elem, self.ir.Constant(i32, 1), f"wg_qeb2{i}_{k}")  # *2 fp16
                    b_off = IRBuilder.add(b_base, eb2, f"sb{i}_{k}_off")
                elif dq_kread:
                    # bwd dQ (dQ_swapAB=0) B = K from SmemLayoutKt = composition(
                    # SmemLayoutK, transpose), narrow (SW32, ATOM=16) atom (hd96).
                    # GEMM roles: contraction kc = key, output n_val = hdim. cute
                    # Kt(hd, key) = (hd%ATOM) + (hd//ATOM)*(kBlockN*ATOM) + key*ATOM,
                    # then sw1 (probe_dqk2_h96 EXACT, 12288/12288). So with kc=key,
                    # n_val=hd:  e = (n_val%ATOM) + (n_val//ATOM)*(kBlockN*ATOM)
                    #                + kc*ATOM, swizzle sw_B. ATOM = 8<<swiz_B_b.
                    #
                    # THE hd-atom stride MUST be the K tile's full seqlen extent
                    # kBlockN*ATOM (= 128*16 = 2048 for hd96). The `kblockn` variable in
                    # scope is `_fa_kblockn`, which is UNRELIABLE here: hd96's bwd has NO
                    # `.tnspA` HGMMA, so `_fa_bwd` is False and the fwd `_fa_kblockn`
                    # capture overwrites it with the LAST SS GEMM's N = the dQ GEMM's
                    # N=48 -> stride 48*16=768, which DOUBLE-UNDER-COUNTS the hd-atom and
                    # leaves the read hd-ATOM (n_val>>4) wrong for ~54% (hin^hd ∈ {16,32,
                    # 48} = pure hd-atom error; the within-atom hd low-4 bits and the key
                    # were 100% correct — KID-injection showed b_kg0==0 always and
                    # b_base carries only the key-block,
                    # so the hd-atom must come from this jump with stride kBlockN*ATOM).
                    # Use `_fa_kblockn_dqk` = (dQ GEMM accum-HGMMA count)×k_dim = the true
                    # kBlockN (128 for hd96), captured in the per-function pre-scan.
                    # Gated dq_kread (dq_pds_a ∧ swiz_B_b<3) = uniquely the hd96 dQ K-read,
                    # so all SW128 (swiz_B_b==3) SS reads stay byte-identical.
                    _ATOM_b = 8 << swiz_B_b
                    _bmask = (1 << swiz_B_b) - 1
                    _kbn_dqk = getattr(self, "_fa_kblockn_dqk", None) or kblockn
                    _n_in = IRBuilder.and_(n_val, self.ir.Constant(i32, _ATOM_b - 1), f"wg_dqk_nin{i}_{k}")
                    _n_atom = IRBuilder.lshr(n_val, self.ir.Constant(i32, _ATOM_b.bit_length() - 1), f"wg_dqk_na{i}_{k}")
                    _hd_jump = IRBuilder.mul(_n_atom, self.ir.Constant(i32, _kbn_dqk * _ATOM_b), f"wg_dqk_hj{i}_{k}")
                    _kc_term = IRBuilder.mul(kc, self.ir.Constant(i32, _ATOM_b), f"wg_dqk_kc{i}_{k}")
                    e_b = IRBuilder.add(IRBuilder.add(_n_in, _hd_jump, f"wg_dqk_e0{i}_{k}"),
                                        _kc_term, f"wg_dqk_e{i}_{k}")
                    yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 7), f"wg_dqk_ys{i}_{k}"),
                                         self.ir.Constant(i32, _bmask), f"wg_dqk_yyy{i}_{k}")
                    es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, 4), f"wg_dqk_xor{i}_{k}"),
                                       f"wg_dqk_es{i}_{k}")
                    eb2 = IRBuilder.shl(es, self.ir.Constant(i32, 1), f"wg_dqk_eb{i}_{k}")  # *2 fp16
                    b_off = IRBuilder.add(b_base, eb2, f"sb{i}_{k}_off")
                elif is_mn_b:
                    # MN-major V: swizzle on (k_local*ATOM_b + n_in), atom on n (hdim).
                    # ATOM_b = 8<<swiz_B_b (SW128 64, default). swiz_B_b default 3 so
                    # every fp16/bf16 fwd PV (all SW128) stays byte-identical; a
                    # narrower-V fwd kernel (e.g. hd96 fwd) reads its SW64 V here.
                    _ATOM_b = 128 if is_fp8 else (8 << swiz_B_b)
                    _bmask = (1 << swiz_B_b) - 1
                    # hd96 fwd SS-form PV V-read (_fwd_vg_narrow): the paired A=P read
                    # applies the fix#2 key un-scramble `key ^ (((key>>3)&1)*3)` (the QK
                    # C-fragment spill reverses the low 2 key bits within each odd 8-block).
                    # The V-read's contraction key MUST use the SAME logical ordering or the
                    # PV GEMM sums P[q,key]·V[key^3,hd] on every bit3-set key -> half of each
                    # 8-block mis-paired -> a uniform ~7% attention blur (corr 0.93). KSTAR
                    # single-key probe (Q=ones,V=rowid) was O=V[ks^3] for bit3(ks) set,
                    # O=V[ks] else -> EXACT XOR-3 misalignment. bit3 of the full key ==
                    # bit3(kc) (b_base carries the *16 block, a multiple of 16), and kc is a
                    # lift-time int here, so fold the un-scramble at lift time. Gated
                    # identically to the P-read fix so SW128 fwd / bwd / fp8 stay byte-id.
                    _kc_v = kc
                    if _fwd_vg_narrow:
                        _kv = k ^ (((k >> 3) & 1) * 3)
                        _kc_v = self.ir.Constant(i32, _kv)
                    e_b = IRBuilder.add(IRBuilder.mul(_kc_v, self.ir.Constant(i32, _ATOM_b), f"wg_kb64_{i}_{k}"),
                                        n_in, f"wg_eb{i}_{k}")
                    if _fwd_vg_narrow:
                        # cp.async-gathered SW64 V (hd96 fwd): COMPACT swizzle
                        # Swizzle<B,B+1,3> (yyy>>(B+4), xor<<(B+1)) and NO PackGQA
                        # (kc&9)<<3 extra term. Decoded exact vs the gather (20480/20480).
                        _sh = swiz_B_b + 4
                        _xs = swiz_B_b + 1
                        yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, _sh), f"wg_ys{i}_{k}"),
                                             self.ir.Constant(i32, _bmask), f"wg_yyy{i}_{k}")
                        es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, _xs), f"wg_xor{i}_{k}"),
                                           f"wg_es{i}_{k}")
                    else:
                        yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 7), f"wg_ys{i}_{k}"),
                                             self.ir.Constant(i32, _bmask), f"wg_yyy{i}_{k}")
                        es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, 4), f"wg_xor{i}_{k}"),
                                           f"wg_es{i}_{k}")
                        if _ersb:
                            rsw = IRBuilder.shl(IRBuilder.and_(kc, self.ir.Constant(i32, 9), f"wg_vrsm{i}_{k}"),
                                                self.ir.Constant(i32, 3), f"wg_vrsw{i}_{k}")
                            es = IRBuilder.xor(es, rsw, f"wg_ves{i}_{k}")
                    elem = IRBuilder.add(atom_off, es, f"wg_belem{i}_{k}")
                    eb2 = IRBuilder.shl(elem, self.ir.Constant(i32, 1), f"wg_eb2{i}_{k}")  # *2 fp16
                    b_off = IRBuilder.add(b_base, eb2, f"sb{i}_{k}_off")
                elif is_fp8 and is_rs:
                    # FP8 PV: B = Vt, the in-kernel TRANSPOSED V tile (LDSM.MT88 +
                    # STSM). The in-kernel transpose does NOT produce cute's plain
                    # SmemLayoutVtMma `n*32 + k_local` layout — decoded exactly
                    # against the runtime transpose output (4096/4096): within a
                    # 32-seqlen block the element index is
                    #     e = sw1( k_local*32 + P(hdim) )
                    # where k_local is the key (contraction) and P(hdim) is a LINEAR
                    # bit map of the hdim output index (the cute tile_to_shape
                    # Step<_1,_2> hdim split, key-major, interleaved with the swizzle):
                    #     P = XOR over set bits i of hdim of:
                    #       [1024, 2048, 1, 1026, 2052, 8, 16]   (bit 0..6)
                    # and sw1(e) = e ^ (((e>>7)&1)<<4) is the 1-bit Swizzle<1,4,3>.
                    # The old `n*32 + k_local` formula had hdim/key effectively
                    # swapped -> every PV B-read landed on the wrong V byte (V=ones
                    # blind; real-V corr ~0). The b_base steps 4096 per 32-key block
                    # (the descriptor start_address), so k here is the in-block key.
                    #
                    # n_val (the GMMA C/D-fragment n-coordinate from the HGMMA-derived
                    # coords_v) is NOT the QGMMA output hdim: the fp8 QGMMA C-layout
                    # rotates the low 4 hdim bits by one vs HGMMA. The O epilogue
                    # stores the accumulator at the QGMMA hdim, so the V read must use
                    # that hdim too. Decoded exactly (best-permutation correlation
                    # 0.99999 of lifted O versus reference): within each 16-hdim block the
                    # true hdim = ROTATE-LEFT-1 of n_val's low 4 bits:
                    #   hd_eff = (n_val & ~15) | ((n_val<<1)&15) | ((n_val>>3)&1)
                    # (gated on fp8 PV; HGMMA/bf16/fp16/bwd never hit this branch).
                    nlo = IRBuilder.and_(n_val, self.ir.Constant(i32, 15), f"wg_vnlo_{i}_{k}")
                    rot = IRBuilder.or_(
                        IRBuilder.and_(IRBuilder.shl(nlo, self.ir.Constant(i32, 1), f"wg_vnsh_{i}_{k}"),
                                       self.ir.Constant(i32, 15), f"wg_vnshm_{i}_{k}"),
                        IRBuilder.lshr(nlo, self.ir.Constant(i32, 3), f"wg_vncarry_{i}_{k}"),
                        f"wg_vnrot_{i}_{k}")
                    hd_eff = IRBuilder.or_(IRBuilder.and_(n_val, self.ir.Constant(i32, ~15 & 0xffffffff), f"wg_vnhi_{i}_{k}"),
                                           rot, f"wg_vhdeff_{i}_{k}")
                    # The Vt smem LAYOUT depends on how V reached smem: TMA-loaded
                    # (base/packgqa/split) vs cp.async-GATHERED (PagedKVNonTMA). The two
                    # in-kernel transposes produce DIFFERENT (and effectively hd<->key
                    # transposed) Vt layouts:
                    #
                    #  - TMA-V (default, _kv_gather_loaded False): decoded vs the runtime
                    #    transpose output (cf6f0322) — within a 32-key block the element
                    #    index is `sw1( k_local*32 + P(hd_eff) )`, key in the stride-32
                    #    field and hd in the linear P bit-map P_TMA.
                    #  - GATHERED V (_kv_gather_loaded True): the cp.async-gathered V is
                    #    transposed into CUTE's nominal SmemLayoutVtMma. Derived exactly
                    #    against tile_to_shape(ss_smem_selector<K,e4m3,128,kBlockN>,
                    #    (128, kBlockN), Step<_1,_2>); BN=160/224 give the same
                    #    within-block layout, 4096/4096. Within a
                    #    32-key block off(hd,key) = Q(hd) ^ key, with the swizzle already
                    #    folded into Q (so NO separate sw1 here):
                    #        Q = XOR over set bits i of hd of
                    #            [32, 64, 144, 256, 512, 1024, 2048]   (bit 0..6)
                    #    The cf6f0322 `k*32 + P(hd)` form had hd/key transposed for this
                    #    layout -> the gathered PV B-read collapsed hd (V averaged over
                    #    hdim; decisive b_off dump showed every read landing on cute
                    #    cells with hd<->key swapped). b_base steps 4096 per 32-key block
                    #    for BOTH layouts. Gated on _v_cpasync_gathered (V via cp.async,
                    #    LDGSTS >> UTMALDG) so the TMA-V path (base/packgqa/split, where V
                    #    is TMA-loaded) is byte-identical — it never enters this arm.
                    _v_gathered = getattr(self, "_v_cpasync_gathered", False)
                    if _v_gathered:
                        Q_IMG = [32, 64, 144, 256, 512, 1024, 2048]
                        q_hd = self.ir.Constant(i32, 0)
                        for _bit, _img in enumerate(Q_IMG):
                            _b = IRBuilder.and_(
                                IRBuilder.lshr(hd_eff, self.ir.Constant(i32, _bit), f"wg_vqh{_bit}_{i}_{k}"),
                                self.ir.Constant(i32, 1), f"wg_vqhm{_bit}_{i}_{k}")
                            _sel = IRBuilder.mul(_b, self.ir.Constant(i32, _img), f"wg_vqi{_bit}_{i}_{k}")
                            q_hd = IRBuilder.xor(q_hd, _sel, f"wg_vq{_bit}_{i}_{k}")
                        # off(hd,key) = Q(hd) ^ key (swizzle folded into Q); 1 byte/elem.
                        es = IRBuilder.xor(q_hd, kc, f"wg_ves8g_{i}_{k}")
                        b_off = IRBuilder.add(b_base, es, f"sb{i}_{k}_off")
                    else:
                        P_IMG = [1024, 2048, 1, 1026, 2052, 8, 16]
                        p_hd = self.ir.Constant(i32, 0)
                        for _bit, _img in enumerate(P_IMG):
                            _b = IRBuilder.and_(
                                IRBuilder.lshr(hd_eff, self.ir.Constant(i32, _bit), f"wg_vph{_bit}_{i}_{k}"),
                                self.ir.Constant(i32, 1), f"wg_vphm{_bit}_{i}_{k}")
                            _sel = IRBuilder.mul(_b, self.ir.Constant(i32, _img), f"wg_vpi{_bit}_{i}_{k}")
                            p_hd = IRBuilder.xor(p_hd, _sel, f"wg_vp{_bit}_{i}_{k}")
                        e_b = IRBuilder.add(IRBuilder.mul(kc, self.ir.Constant(i32, 32), f"wg_vk32_{i}_{k}"),
                                            p_hd, f"wg_veb{i}_{k}")
                        sb = IRBuilder.shl(IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 7), f"wg_vs{i}_{k}"),
                                                          self.ir.Constant(i32, 1), f"wg_vsm{i}_{k}"),
                                           self.ir.Constant(i32, 4), f"wg_vsh{i}_{k}")
                        es = IRBuilder.xor(e_b, sb, f"wg_ves8_{i}_{k}")   # 1 byte/elem fp8
                        b_off = IRBuilder.add(b_base, es, f"sb{i}_{k}_off")
                    if os.environ.get("SLIFTER_PV_BOFF_DBG"):
                        self.run_intrinsic("slifter_pv_boff_dbg", IRBuilder,
                                           [n_val, kc, b_off], f"pvb{i}_{k}")
                elif dq_pds_b:
                    # bwd dQ B = dS read from the STSM-written sdS tile. b_base carries
                    # the key-block second-atom jump, so kg_b is the local key.
                    kg_b = IRBuilder.add(b_kg0, kc, f"wg_kgb{i}_{k}")
                    _pds_atom = getattr(self, "_fa_pds_atom", None) or 64
                    if _pds_atom > 32 and _pds_atom <= 48:
                        # 40-wide PdS atom (hd256): PLAIN LINEAR, NO swizzle (same tile
                        # the dkv_pdst_b atom-40 branch decoded; SLIFTER_STSM_FRAG WG0
                        # 255/256, no Sw<1>/bit-coupling):
                        #   e = (key>>3)*512 + query*8 + (key&7)
                        # query = n_val (dQ output row), key = kg_b (the contraction key).
                        # Same second-warpgroup key-pair swap as dkv_pdst_b: keys >=
                        # kBlockN/2 (the WG1 sub-tile) store (even,odd) pairs swapped, so
                        # XOR the key low bit there. kblockn = _fa_kblockn_bwd = 80.
                        khalf = self.ir.Constant(i32, kblockn >> 1)
                        _is_wg1 = IRBuilder.icmp_unsigned(">=", kg_b, khalf, f"wg_pdsw1{i}_{k}")
                        _xb = IRBuilder.select(_is_wg1, self.ir.Constant(i32, 1),
                                               self.ir.Constant(i32, 0), f"wg_pdsxb{i}_{k}")
                        nin = IRBuilder.xor(
                            IRBuilder.and_(kg_b, self.ir.Constant(i32, 7), f"wg_pdsn0{i}_{k}"),
                            _xb, f"wg_pdsn{i}_{k}")
                        kblk = IRBuilder.lshr(kg_b, self.ir.Constant(i32, 3), f"wg_pdskb{i}_{k}")
                        es = IRBuilder.add(
                            IRBuilder.add(
                                IRBuilder.mul(n_val, self.ir.Constant(i32, 8), f"wg_pdsq{i}_{k}"),
                                nin, f"wg_pdse0{i}_{k}"),
                            IRBuilder.mul(kblk, self.ir.Constant(i32, 512), f"wg_pdskm{i}_{k}"),
                            f"wg_pdses{i}_{k}")
                    else:
                        # 64-wide atom (hd64/128): SmemLayoutPdS Sw<3,4,3> on
                        # e = query*64 + key_local.
                        e_b = IRBuilder.add(IRBuilder.mul(n_val, self.ir.Constant(i32, 64), f"wg_pdsq{i}_{k}"),
                                            kg_b, f"wg_pdse{i}_{k}")
                        yyy = IRBuilder.and_(IRBuilder.lshr(e_b, self.ir.Constant(i32, 6), f"wg_pdss{i}_{k}"),
                                             self.ir.Constant(i32, 7), f"wg_pdsy{i}_{k}")
                        es = IRBuilder.xor(e_b, IRBuilder.shl(yyy, self.ir.Constant(i32, 3), f"wg_pdsx{i}_{k}"),
                                           f"wg_pdses{i}_{k}")
                    eb2 = IRBuilder.shl(es, self.ir.Constant(i32, 1), f"wg_pdseb{i}_{k}")  # *2 fp16
                    b_off = IRBuilder.add(b_base, eb2, f"sb{i}_{k}_off")
                else:
                    kg_b = IRBuilder.add(b_kg0, kc, f"wg_kgb{i}_{k}")
                    if b_transposed and not is_rs:
                        # SS `.tnspB`: B read MN-major (cute Major::MN) from a K-major
                        # SW128 tile = same physical layout with (row,k) swapped (the
                        # A-side `.tnspA` mirror, proved 0/512). Latent for the current
                        # bwd config (its SS B is plain K-major) but needed for general
                        # SS `.tnspB` forms.
                        b_off = self._sw128_phys_off(IRBuilder, b_base, kg_b, n_val, K_total_b, f"sb{i}_{k}",
                                                     extra_row_swizzle=_ersb,
                                                     byte_per_elem=(1 if is_fp8 else 2),
                                                     swiz_B=swiz_B_b)
                    else:
                        b_off = self._sw128_phys_off(IRBuilder, b_base, n_val, kg_b, K_total_b, f"sb{i}_{k}",
                                                     extra_row_swizzle=_ersb,
                                                     byte_per_elem=(1 if is_fp8 else 2),
                                                     swiz_B=swiz_B_b,
                                                     compact_swizzle=_fwd_kg_narrow)
                if is_rs:
                    # A=P from per-warpgroup scratch: slifter_rs_a_read(m, k_local)
                    # returns the A element in the low bits of an i32 (16-bit for
                    # fp16/bf16, 8-bit e4m3 byte for fp8). fp8 uses its own scratch
                    # (k-stride 32) so route to the fp8 read fn.
                    _read_fn = "slifter_rs_a_read_fp8" if is_fp8 else "slifter_rs_a_read"
                    rd_m, rd_k = m_val, kc
                    if dp_av_sw64:
                        # hd96 dP A=V: the LDSM-loaded V fragment lands SW64-scrambled
                        # in the spill scratch (slot[m,k] holds V[key=m^((m>>3)&1),
                        # hd_in=k^(((m>>1)&1)<<3)]). To read logical V[key=m_val,
                        # hd_in=kc] read the scrambled slot (m,k):
                        #   m = m_val ^ ((m_val>>3)&1) ; k = kc ^ (((m_val>>1)&1)<<3)
                        # (m bit3 / m bit1 are themselves unflipped, so this is a clean
                        # self-consistent inverse; bijection 2048/2048 vid-injected.)
                        _mb3 = IRBuilder.and_(IRBuilder.lshr(m_val, self.ir.Constant(i32, 3), f"rsa_mb3s{i}_{k}"),
                                              self.ir.Constant(i32, 1), f"rsa_mb3{i}_{k}")
                        rd_m = IRBuilder.xor(m_val, _mb3, f"rsa_swm{i}_{k}")
                        _mb1 = IRBuilder.shl(
                            IRBuilder.and_(IRBuilder.lshr(m_val, self.ir.Constant(i32, 1), f"rsa_mb1s{i}_{k}"),
                                           self.ir.Constant(i32, 1), f"rsa_mb1{i}_{k}"),
                            self.ir.Constant(i32, 3), f"rsa_mb1sh{i}_{k}")
                        rd_k = IRBuilder.xor(kc, _mb1, f"rsa_swk{i}_{k}")
                    a_bits = self.run_intrinsic(_read_fn, IRBuilder, [rd_m, rd_k], f"rsa{i}_{k}")
                    if is_fp8:
                        av = self._e4m3_byte_to_f32(IRBuilder, a_bits, f"rsa8_{i}_{k}")
                    elif is_bf16:
                        # bf16 = top 16 bits of an f32: (bits & 0xffff) << 16, bitcast.
                        b32 = IRBuilder.shl(IRBuilder.and_(a_bits, self.ir.Constant(i32, 0xffff), f"rsabf_a{i}_{k}"),
                                            self.ir.Constant(i32, 16), f"rsabf_s{i}_{k}")
                        av = IRBuilder.bitcast(b32, f32, f"rsaf_{i}_{k}")
                    else:
                        a_h = IRBuilder.bitcast(IRBuilder.trunc(a_bits, i16, f"rsa16_{i}_{k}"), half, f"rsah_{i}_{k}")
                        av = IRBuilder.fpext(a_h, f32, f"rsaf_{i}_{k}")
                elif is_fp8:
                    av = self._shared_8_e4m3_load(IRBuilder, a_off, f"wg_a{i}_{k}")
                else:
                    av = self._shared_16_load(IRBuilder, a_off, f"wg_a{i}_{k}", is_bf16)
                if is_fp8:
                    bv = self._shared_8_e4m3_load(IRBuilder, b_off, f"wg_b{i}_{k}")
                else:
                    bv = self._shared_16_load(IRBuilder, b_off, f"wg_b{i}_{k}", is_bf16)
                if is_rs and os.environ.get("SLIFTER_PVOP_DUMP"):
                    self.run_intrinsic("slifter_pvop_dump", IRBuilder,
                                       [m_val, n_val, kc, b_base, av, bv], f"pvop{i}_{k}")
                # DEBUG (gated lift-time on dq_kread + runtime SLIFTER_KID_INJECT): dump
                # the hd96 dQ K-read (n_val=hd, kc=key-within-MMA, b_base, b_off, bv).
                # With K smem KID-injected, bv decodes to the (key,hd) the read landed on
                # -> NOISE-FREE verification that read_hd == n_val (hd-atom correct).
                if dq_kread and os.environ.get("SLIFTER_KID_INJECT"):
                    self.run_intrinsic("slifter_dqk_dump", IRBuilder,
                                       [n_val, kc, b_base, b_kg0, b_off, bv], f"dqk{i}_{k}")
                # DEBUG (gated): dump SS-form operands (S/dP/dQ GEMMs) — a_base/b_base
                # and the per-k av/bv operand — to pin whether the swapAB SS read
                # resolves to the K/Q smem the TMA wrote (suspect b) or reads zero
                # (suspect c). Keyed on a_base/b_base lets the analyzer isolate the S
                # GEMM (a_base = Q region, b_base = K region ~0x1a000>>4 byte).
                # SLIFTER_SOP_DUMP: all SS GEMMs (S/dP/dQ). SLIFTER_SOP_DUMP_DQ:
                # ONLY the dQ GEMM (a_transposed=.tnspA) — far smaller volume so the
                # dump reaches dQ without the giant-function overrun. Both lift-time
                # gated (prod IR untouched when neither env is set at lift time).
                _sop = (os.environ.get("SLIFTER_SOP_DUMP")
                        or (os.environ.get("SLIFTER_SOP_DUMP_DQ") and a_transposed))
                if not is_rs and _sop:
                    # Record the COMPUTED smem byte offsets a_off/b_off (not the bases)
                    # so the analyzer can pin exactly which (key,hd)/(query,hd) cell the
                    # A/B reads land on vs where the TMA wrote K/Q.
                    self.run_intrinsic("slifter_sop_dump", IRBuilder,
                                       [m_val, n_val, kc, a_off, b_off, av, bv], f"sop{i}_{k}")
                prod = IRBuilder.fmul(av, bv, f"wg_p{i}_{k}")
                acc = IRBuilder.fadd(acc, prod, f"wg_acc{i}_{k}")
            # DEBUG (gated env, additive call only — no production behavior change):
            # dump the QK S fragment (SS form) for element-wise compare vs a CPU
            # scale*Q.K reference.
            if not is_rs and os.environ.get("SLIFTER_S_DUMP"):
                self.run_intrinsic("slifter_s_dump", IRBuilder, [m_val, n_val, a_base, b_base, acc], f"sdump{i}")
            # DEBUG (gated): dump the PV (is_rs) acc per (m,n) to see whether the
            # decode produces nonzero acc for the hdim blocks that end up 0 in O.
            if is_rs and os.environ.get("SLIFTER_PVACC_DUMP"):
                self.run_intrinsic("slifter_pvacc_dump", IRBuilder, [m_val, n_val, acc], f"pvacc{i}")
            self._set_val(IRBuilder, IRRegs, d, acc)
        return True

    def _try_wgmma_descriptor_product(self, Inst, desc_op, desc_hi_op, a_regs, c_regs, defs,
                                      m_dim, n_dim, k_dim, IRBuilder, IRRegs, _get_val,
                                      desc_a_hi_op=None,
                                      is_bf16=False, is_fp8=False,
                                      a_transposed=False, b_transposed=False):
        """Compute the genuine A*B product for a descriptor-form HGMMA/WGMMA by
        reading the A and B fragments back out of @shared_mem.

        Decodes the GMMA shared-memory matrix descriptor for TWO K-major layouts:
          - no-swizzle K-INTER (cute Layout_K_INTER_Atom): single HGMMA, full k.
          - 128B-swizzle SW128 (cute Layout_K_SW128_Atom): FlashAttention's real
            Q/K/V layout. The K_total-wide tile is one TMA load; cute issues
            K_total/16 accumulating HGMMAs, the descriptor low word stepping +2
            (= +16 k-elements at row 0 = +32 bytes) per HGMMA. The physical smem
            offset of (m,k) is the row-major linear index swizzled on the element
            index (probed exactly vs cute): sw128(m*K_total + k).
        Both are fp16-in / fp32-accum, single-tile single-CTA. On x86 the wrapper
        runs all 128 warpgroup threads; each owns a distinct D-fragment (cute
        CLayout_64xN) computed from SR_TID.X at runtime.

        Returns True if it emitted the product; False to fall back to passthrough.

        Restrictions (anything else → return False, keep C-passthrough):
          - k_dim == 16, m_dim == 64
          - F32 accumulator, d_count == n_dim/2 == len(defs)
          - descriptor register pair (lo=A-base, hi=B-base) resolvable in IRRegs
        """
        # fp16/bf16 HGMMA: k_dim==16. fp8 QGMMA: k_dim==32 (1-byte elements pack
        # twice as many k per MMA). Both are m64.
        expected_k = 32 if is_fp8 else 16
        if k_dim != expected_k or m_dim != 64:
            return False
        d_count = max(n_dim // 2, 4)
        if len(defs) != d_count:
            return False

        i32 = self.ir.IntType(32)
        i32t = i32
        f32 = self.ir.FloatType()

        # --- Detect SW128 vs K-INTER from desc_a's HIGH word (layout_type bits
        # [63:62] = high-word bits[31:30]; B128/SW128 == 1). The parser carries
        # UR(n+1) tagged desc_a_hi so we can read it; if absent, assume K-INTER.
        is_sw128 = False
        if desc_a_hi_op is not None:
            try:
                ahi = _get_val(desc_a_hi_op, "wgmma_descAhi")
                # The descriptor HIGH word carries layout_type in bits[31:30]
                # (== desc bits[63:62]; SW128/B128 == 1). It reaches the lifter
                # typed as float and, for the 2nd..Nth accumulating HGMMA, wrapped
                # in a UIADD3 carry chain (the SASS rebuilds UR(n+1) via
                # `UIADD3.X URZ, 0x40000040, ... + carry`). The +small low-word
                # carry can never reach bit 30, so the layout bit equals that of
                # the 0x40000040 immediate. _layout_bits_30_31 peels bitcast/ext
                # AND add/or operands to find that immediate's layout bits.
                lt = self._layout_bits_30_31(ahi)
                if lt is not None:
                    is_sw128 = (lt == 1)
            except Exception:
                is_sw128 = False

        # RecordWgmmaSwizzle stashes the per-operand swizzle bit-width B (3/2/1 =
        # SW128/SW64/SW32) read statically from the descriptor HIGH-word layout
        # immediate (per-GEMM, handling UR reuse). ANY recorded swizzle (B>=1) on
        # either operand means the tile IS swizzled and must take the (generalized)
        # _wgmma_sw128_product path — not the no-swizzle K-INTER branch. The
        # runtime _layout_bits probe above only flags lt==1 (SW128); hd96's K is
        # SW32 (lt==3) so the probe leaves is_sw128 False and the QK GEMM would
        # wrongly fall to K-INTER (no swizzle). This static check fixes that while
        # staying byte-identical for SW128 kernels (they already set is_sw128 via
        # the probe / a_regs / fp8 / ublkred gates below; B=3 there reproduces the
        # exact SW128 swizzle).
        _rec_Ba = getattr(Inst, "_swiz_B_a", None)
        _rec_Bb = getattr(Inst, "_swiz_B_b", None)
        if (_rec_Ba is not None and _rec_Ba >= 1) or (_rec_Bb is not None and _rec_Bb >= 1):
            is_sw128 = True

        # RS-form (a_regs present => A read from registers, e.g. FA3's PV=P@V)
        # MUST take the SW128 path: only _wgmma_sw128_product reads the A operand
        # from the spilled per-warpgroup register scratch (slifter_rs_a_read); the
        # K-INTER branch reads A from shared memory, where an RS-form A is NOT
        # stored -> garbage A. The desc_a_hi layout-word probe above is moreover
        # unreliable for RS-form: the parser reads desc_a_hi as UR(lo+1), but for
        # some PV sites that UR is reused (e.g. the FA3 split epilogue PV's UR5
        # holds the V-wait mbarrier address, not the 0x40000040 SW128 layout word),
        # so is_sw128 wrongly reads False. Force SW128 for RS-form. SS-form
        # (a_regs empty: the QK GEMM and all WGMMA microbench tests) is unaffected.
        if a_regs:
            is_sw128 = True

        # FP8 (QGMMA) FlashAttention always uses the SW128 Q/K/V layout (cute
        # Layout_K_SW128_Atom<float_e4m3_t>, K-atom = 128 elements). Force SW128
        # so the 8-bit e4m3 decode path runs (the desc_a_hi layout probe is as
        # unreliable for fp8 as for RS-form).
        if is_fp8:
            is_sw128 = True

        # FA backward SS-form transposed GEMMs (dQ `.tnspB`-only; dKV
        # `.tnspA.tnspB`) MUST take the SW128 path so the PdS dS A-read (dq_pds_a)
        # and the PdSt dS B-read (dkv_pdst_b) inside _wgmma_sw128_product fire. The
        # desc_a_hi layout-word probe is unreliable for these (the bwd rebuilds the
        # descriptor HIGH word via a UIADD3 carry chain whose immediate the probe
        # may not resolve for the dQ `.tnspB`-only GEMM, leaving is_sw128 False ->
        # the K-INTER branch reads A from the wrong smem). Gated on UBLKRED (bwd
        # mainloop only) + a transpose flag (the QK/SdP plain SS GEMMs have no tnsp
        # and keep the probe path) + SS-form (a_regs empty). fwd has no UBLKRED.
        if (getattr(self, "_has_ublkred", False) and not a_regs
                and (a_transposed or b_transposed)):
            is_sw128 = True

        if is_sw128:
            # Per-operand swizzle width from the descriptor HIGH words. A's mode is
            # in desc_a_hi (UR n+1); B's in desc_b_hi (UR n+3). Both default to 3
            # (SW128) when unresolvable, so SW128 kernels (hd128/192/256 + all fwd)
            # are byte-identical. hd96's mixed atoms (Q/dO SW64 B=2, K SW32 B=1,
            # V SW64 B=2) are read here. fp8 keeps the fixed 128-wide e4m3 atom
            # (its own decode path); the swizzle-B parametrization is fp16/bf16.
            # Per-operand swizzle bit-width recorded pre-rename by
            # RecordWgmmaSwizzle (A.hi UR n+1, B.hi UR n+3). Default 3 (SW128) when
            # unrecorded so SW128 kernels (hd128/192/256 + all fwd) are byte-
            # identical; fp8 keeps its fixed 128-wide e4m3 atom path.
            swiz_B_a = 3
            swiz_B_b = 3
            if not is_fp8:
                swiz_B_a = int(getattr(Inst, "_swiz_B_a", 3))
                swiz_B_b = int(getattr(Inst, "_swiz_B_b", 3))
            return self._wgmma_sw128_product(
                desc_op, desc_hi_op, c_regs, defs, n_dim, k_dim,
                IRBuilder, IRRegs, _get_val, a_regs=a_regs, is_bf16=is_bf16,
                is_fp8=is_fp8, a_transposed=a_transposed, b_transposed=b_transposed,
                swiz_B_a=swiz_B_a, swiz_B_b=swiz_B_b)

        # Descriptor pair: lo (desc_op) = A descriptor, hi (desc_hi_op) = B
        # descriptor. Each descriptor's low 14 bits hold start_address =
        # (shared byte offset >> 4). The parser now emits BOTH the LO operand
        # (named on the SASS gdesc[URn]) and the synthesized HI operand UR(n+1),
        # so both A's and B's address-build chains survive DCE and are
        # materialized in IRRegs (after RegRemap the URs become ordinary R regs).
        # Fetch both through the normal operand path.
        if desc_hi_op is None:
            return False
        lo_val = None
        hi_val = None
        try:
            lo_val = _get_val(desc_op, "wgmma_descA")
            hi_val = _get_val(desc_hi_op, "wgmma_descB")
        except Exception:
            return False
        if lo_val is None or hi_val is None:
            return False
        # Both values are i32 (descriptor low words). Normalize to i32.
        i32t = self.ir.IntType(32)
        if lo_val.type != i32t:
            lo_val = self._insert_type_cast_if_needed(IRBuilder, lo_val, i32t, "wgmma_descA_i32")
        if hi_val.type != i32t:
            hi_val = self._insert_type_cast_if_needed(IRBuilder, hi_val, i32t, "wgmma_descB_i32")

        i32 = self.ir.IntType(32)
        f32 = self.ir.FloatType()

        def start_byte(v, nm):
            # descriptor reg holds (smem_start_byte >> 4) | layout-bits.
            # start_address is the low 14 bits; recover the byte offset (<<4).
            m = IRBuilder.and_(v, self.ir.Constant(i32, 0x3fff), f"{nm}_sa")
            return IRBuilder.shl(m, self.ir.Constant(i32, 4), f"{nm}_byte")

        a_base = start_byte(lo_val, "wg_a")   # A shared byte offset
        b_base = start_byte(hi_val, "wg_b")   # B shared byte offset

        # Physical offset (in fp16 elements) of logical (row,k) in the canonical
        # no-swizzle K-INTER layout:  off = row*8 + (k & 7) + (k>>3)*ROWS*8
        # (cute: ((R,(8,2)):(8,(1, R*8))) — leading 8 fp16, K split into 8+8).
        # row is now a runtime i32 IR value (it depends on the executing thread).
        def phys_off(base_bytes, row_val, k_val, rows, nm):
            k_lo = IRBuilder.and_(k_val, self.ir.Constant(i32, 7), f"{nm}_klo")
            k_hi = IRBuilder.lshr(k_val, self.ir.Constant(i32, 3), f"{nm}_khi")
            blk = IRBuilder.mul(k_hi, self.ir.Constant(i32, rows * 8), f"{nm}_blk")
            row8 = IRBuilder.mul(row_val, self.ir.Constant(i32, 8), f"{nm}_row8")
            elem = IRBuilder.add(row8,
                                 IRBuilder.add(k_lo, blk, f"{nm}_e0"), f"{nm}_elem")
            # byte offset = base + elem*2 (fp16)
            eb = IRBuilder.shl(elem, self.ir.Constant(i32, 1), f"{nm}_eb")
            return IRBuilder.add(base_bytes, eb, f"{nm}_off")

        # Per-thread D-fragment coordinates from cute's CLayout_64xN:
        #   Layout<Shape <Shape <  4, 8, 4>, Shape < 2, 2, n_dim/8>>,
        #          Stride<Stride<128, 1,16>, Stride<64, 8, 512>>>
        # The flattened index is m + 64*n (C is M=64 major). Decoding the
        # thread mode t = (t0:4, t1:8, t2:4) and value mode v = (v0:2, v1:2, v2):
        #   m = t1 + 16*t2 + 8*v1          (t1=(t/4)%8 row-in-warp, t2=t/32 warp)
        #   n = 2*t0 + v0 + 8*v2           (t0=t%4)
        # This is the EXACT coordinate each of the 128 warpgroup threads owns,
        # so every thread's STG writes the correct D location (not just lane 0).
        tid = self._load_const_mem_u32(IRBuilder, 0x2C, "wgmma_tid")  # SR_TID.X
        t0 = IRBuilder.and_(tid, self.ir.Constant(i32, 3), "wg_t0")
        t1 = IRBuilder.and_(IRBuilder.lshr(tid, self.ir.Constant(i32, 2), "wg_t1s"),
                            self.ir.Constant(i32, 7), "wg_t1")
        # warpgroup-LOCAL warp index (0..3). SR_TID.X is block-global, so a warp-
        # specialized kernel's consumer warpgroups have tid>=128; without &3 the warp
        # index is 4..11 and row m=t1+16*t2 lands >=64, outside the 64-row WGMMA tile
        # (FA3 consumers then read/store garbage). t0/t1 use bits[0:5] = already local.
        t2 = IRBuilder.and_(IRBuilder.lshr(tid, self.ir.Constant(i32, 5), "wg_t2s"),
                            self.ir.Constant(i32, 3), "wg_t2")
        m_warp = IRBuilder.add(t1, IRBuilder.mul(t2, self.ir.Constant(i32, 16), "wg_mw0"),
                               "wg_mwarp")          # t1 + 16*t2 (no value contribution yet)
        n_lane = IRBuilder.mul(t0, self.ir.Constant(i32, 2), "wg_nlane")  # 2*t0

        # Per-value (v) constant pieces: v0=v%2 (-> n), v1=(v/2)%2 (-> m), v2=v/4 (-> n)
        n_oct = max(n_dim // 8, 1)
        coords_v = []
        for v in range(d_count):
            v0 = v % 2
            v1 = (v // 2) % 2
            v2 = v // 4
            coords_v.append((v1, v0, v2))   # (m-add/8, n-add, n-add/8)
        if len(coords_v) != d_count or n_oct * 4 != d_count:
            return False

        # Emit, per D register, the K-reduction dot product D[m,n]=sum_k A[m,k]*B[n,k]
        # plus the C accumulator (RZ -> 0). m,n are runtime (tid-dependent).
        for i, d in enumerate(defs):
            v1, v0, v2 = coords_v[i]
            m_val = IRBuilder.add(m_warp, self.ir.Constant(i32, 8 * v1), f"wg_m{i}")
            n_val = IRBuilder.add(n_lane, self.ir.Constant(i32, v0 + 8 * v2), f"wg_n{i}")
            acc = self.ir.Constant(f32, 0.0)
            if i < len(c_regs):
                acc = _get_val(c_regs[i], f"wgmma_c{i}")
                if acc.type != f32:
                    acc = IRBuilder.bitcast(acc, f32, f"wgmma_c{i}_f32")
            for k in range(k_dim):
                kc = self.ir.Constant(i32, k)
                a_off = phys_off(a_base, m_val, kc, 64, f"a{i}_{k}")
                b_off = phys_off(b_base, n_val, kc, n_dim, f"b{i}_{k}")
                av = self._shared_half_load(IRBuilder, a_off, f"wg_a{i}_{k}")
                bv = self._shared_half_load(IRBuilder, b_off, f"wg_b{i}_{k}")
                prod = IRBuilder.fmul(av, bv, f"wg_p{i}_{k}")
                acc = IRBuilder.fadd(acc, prod, f"wg_acc{i}_{k}")
            self._set_val(IRBuilder, IRRegs, d, acc)
        return True

    @lift_for("HGMMA", "WGMMA", "QGMMA")
    def lift_wgmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """HGMMA/WGMMA/QGMMA: warpgroup MMA (SM90).

        QGMMA is the FP8 (e4m3/e5m2) variant: 1-byte input elements, K-step 32,
        F32 accumulator. The SS/RS descriptor operand layout is identical to
        HGMMA; only the smem element decode differs (8-bit byte -> e4m3 -> f32)
        and the SW128 K-atom is 128 elements wide (vs 64 for fp16). Detected by
        the ".E4M3"/".E5M2" opcode tokens and threaded through as is_fp8.

        Supported forms on x86:
          1. Register form (m64nNk16 F32, N in {8,16,32,64,128,256}):
             scalar FMA decomposition (lane 0, warp 0)
          2. Descriptor form (any shape): pass through C to D
             (descriptor data unresolvable on x86)
          3. RZ dest (barrier form): NOP

        Unsupported forms raise UnsupportedInstructionException.
        """
        shape_str = Inst.opcodes[1] if len(Inst.opcodes) > 1 else ""
        accum_type = Inst.opcodes[2] if len(Inst.opcodes) > 2 else "F32"

        defs = Inst.get_defs()
        uses = Inst.get_uses()

        # Parse shape dimensions
        parts = shape_str.split("x")
        m_dim = int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 64
        n_dim = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 8
        k_dim = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 16

        # Check if D is RZ (barrier/NOP form)
        if all(d.reg == "RZ" for d in defs):
            return  # NOP — barrier/fence form

        # Detect descriptor vs register form
        # Check both name (for parsed gdesc operands) and desc_reg (set by parser)
        has_descriptor = any(
            (hasattr(u, 'name') and u.name and 'gdesc' in str(u.name)) or
            (hasattr(u, 'desc_reg') and u.desc_reg is not None)
            for u in uses
        )

        if has_descriptor:
            # Find the descriptor operand (holds the A shared-mem base reg). The
            # parser tags the LO descriptor with desc_reg/gdesc name and, for the
            # SS form, appends a HI descriptor operand (desc_hi=True) naming the
            # consecutive UR that holds the B shared-mem base.
            desc_op = None
            desc_hi_op = None
            desc_a_hi_op = None
            for u in uses:
                if getattr(u, 'desc_a_hi', False):
                    desc_a_hi_op = u
                    continue
                if getattr(u, 'desc_hi', False):
                    desc_hi_op = u
                    continue
                if desc_op is None and (
                        (hasattr(u, 'name') and u.name and 'gdesc' in str(u.name)) or
                        getattr(u, 'desc_reg', None) is not None):
                    desc_op = u
            # Register-source-A operands (RS form) sit between the D group and the
            # descriptor in the expanded operand list; collect them for the decode.
            a_reg_ops = [u for u in uses if u.is_reg and u is not desc_op
                          and u is not desc_hi_op and u is not desc_a_hi_op
                          and not (hasattr(u, 'name') and u.name and 'gdesc' in str(u.name))
                          and getattr(u, 'desc_reg', None) is None
                          and u.reg != "RZ"
                          and not (hasattr(u, 'name') and u.name and u.name.startswith("gsb"))]
            # The C accumulator group is the trailing register run; but in the
            # operand layout C follows the descriptor. Distinguish A regs (RS,
            # before desc) from C regs by position relative to desc_op.
            try:
                desc_pos = uses.index(desc_op) if desc_op is not None else -1
            except ValueError:
                desc_pos = -1
            a_regs = [u for u in a_reg_ops if uses.index(u) < desc_pos] if desc_pos >= 0 else []
            c_regs = [u for u in a_reg_ops if uses.index(u) > desc_pos] if desc_pos >= 0 else a_reg_ops

            # Attempt a real shared-memory descriptor decode so the lifted kernel
            # computes the genuine A*B product instead of passing C through. This
            # is what makes SM90 tensor-core kernels (CUTLASS / FlashAttention)
            # numerically verifiable on x86. Only the canonical no-swizzle K-major
            # F32 case is decoded; anything else falls back to C-passthrough.
            # bf16-input HGMMA carries a distinct ".BF16" opcode token
            # (HGMMA.64x128x16.F32.BF16) vs fp16's HGMMA.64x128x16.F32; dispatch
            # the smem A/B element decode on it (bf16 = f32 high 16 bits).
            is_bf16 = "BF16" in Inst.opcodes
            # FP8 (QGMMA): 1-byte e4m3/e5m2 smem inputs. Decode each smem byte as
            # e4m3 -> f32 in the descriptor product; descale is folded into the
            # SASS softmax/finalize so the GEMM reads the raw fp8 product.
            is_fp8 = "E4M3" in Inst.opcodes or "E5M2" in Inst.opcodes
            # Transpose flags: the WGMMA descriptor operand carries `.tnspA`/`.tnspB`
            # (cute GMMA::Major::MN read of a K-major smem tile) in its name/suffix.
            # `.tnspA` => the A operand is read MN-major (the SwapAB SdP/dQ GEMMs);
            # `.tnspB` => the B operand is read MN-major (fwd PV V, bwd dKV dO/Q).
            def _tflag(op, tok):
                if op is None:
                    return False
                return tok in str(getattr(op, "name", "") or "") or \
                       tok in str(getattr(op, "suffix", "") or "")
            a_transposed = _tflag(desc_op, "tnspA")
            b_transposed = _tflag(desc_op, "tnspB") or _tflag(desc_hi_op, "tnspB")
            if accum_type.startswith("F32") and self._try_wgmma_descriptor_product(
                    Inst, desc_op, desc_hi_op, a_regs, c_regs, defs,
                    m_dim, n_dim, k_dim, IRBuilder, IRRegs, _get_val,
                    desc_a_hi_op=desc_a_hi_op, is_bf16=is_bf16, is_fp8=is_fp8,
                    a_transposed=a_transposed, b_transposed=b_transposed):
                return

            # Fallback: pass through C to D (descriptor layout unresolved).
            for i, d in enumerate(defs):
                if i < len(c_regs):
                    c_val = _get_val(c_regs[i], f"wgmma_c{i}")
                    self._set_val(IRBuilder, IRRegs, d, c_val)
                else:
                    # No matching C reg — set to zero
                    if accum_type.startswith("F16"):
                        zero = self.ir.Constant(self.ir.IntType(32), 0)
                    else:
                        zero = self.ir.Constant(self.ir.FloatType(), 0.0)
                    self._set_val(IRBuilder, IRRegs, d, zero)
            return

        # Register form: decompose to scalar FMA
        # Supported shapes: M64 x N x K16 where N in {8,16,32,64,128,256}
        if n_dim in (8, 16, 32, 64, 128, 256) and k_dim == 16:
            if accum_type == "F32" or accum_type.startswith("F32"):
                a_count = 4
                b_count = max(n_dim // 4, 2)
                n_fp16_regs = a_count + b_count
                d_count = max(n_dim // 2, 4)
                intrinsic_name = f"wgmma64{n_dim}16f32"

                v2f16 = self.ir.VectorType(self.ir.HalfType(), 2)
                i32 = self.ir.IntType(32)
                f32 = self.ir.FloatType()
                args = []
                for i, u in enumerate(uses):
                    v = _get_val(u, f"wgmma_arg_{i}")
                    target = v2f16 if i < n_fp16_regs else f32
                    try:
                        vbits = v.type.width if v.type.intrinsic_name[0] == 'i' else int(v.type.intrinsic_name[1:])
                    except (AttributeError, ValueError):
                        vbits = 32
                    tbits = 32
                    if vbits != tbits:
                        v = self._insert_type_cast_if_needed(IRBuilder, v, i32, f"wgmma_arg_{i}")
                    if v.type != target:
                        v = IRBuilder.bitcast(v, target, f"wgmma_arg_{i}_cast")
                    args.append(v)
                val = self.run_intrinsic(intrinsic_name, IRBuilder, args, "wgmma_call")
                for i in range(len(defs)):
                    self._set_val(IRBuilder, IRRegs, defs[i], IRBuilder.extract_value(
                        val, i, f"wgmma_res_{i}"
                    ))
                return

        # Unsupported form — fail loudly
        raise UnsupportedInstructionException(
            f"Unsupported WGMMA/HGMMA form: {'.'.join(Inst.opcodes)} "
            f"(shape={shape_str}, accum={accum_type}, n_dim={n_dim}, "
            f"descriptor={has_descriptor})"
        )

    def postprocess_ir(self, ir_code):
        ir_code = super().postprocess_ir(ir_code)
        ir_code = re.sub(
            r'(@"(?:const_mem|local_mem)"\s*=\s*external)\s+global\b',
            r'\1 thread_local global',
            ir_code
        )
        ir_code = re.sub(
            r'(@"const_mem"\s*=)\s+global\b(?!\s*thread_local)',
            r'\1 thread_local global',
            ir_code
        )
        return ir_code
