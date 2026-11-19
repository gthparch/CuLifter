import re
from functools import partial

from llvmlite import binding, ir

from transform.build_cfg import BuildCFG
from transform.handle_call import HandleCall
from transform.resolve_indirect_call import ResolveIndirectCall
from transform.cg_patterns import CGPatterns
from transform.intdiv_recognize import IntDivRecognize
from transform.predicated_to_sel import PredicatedToSel
from transform.sr_substitute_reverse import SRSubstituteReverse
from transform.opmod_transform import OpModTransform
from transform.defuse_analysis import DefUseAnalysis, FullDefUseAnalysis
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
from transform.warp_reconverge import WarpReconverge
from lift.lifter import Lifter, lift_for, UnsupportedInstructionException


def _patch_volatile_memops():
    """llvmlite 0.44's LoadInstr/StoreInstr.descr don't emit `volatile`. Some
    shared-memory warp-synchronous idioms (`volatile __shared__` scans, e.g.
    split's radixSortBlocksKeys) REQUIRE volatile shared loads/stores: otherwise
    `llc -O2` store-to-load-forwards and dead-store-eliminates the per-lane shared
    writes that the next level's cross-lane reads depend on, breaking the scan.
    Honor a `.volatile` flag that lift_lds/lift_sts set on shared LDS/STS so those
    emit `ld.volatile.shared`/`st.volatile.shared` (other mem ops are untouched)."""
    if getattr(ir.LoadInstr, "_slifter_volatile_patched", False):
        return
    _ld = ir.LoadInstr.descr

    def load_descr(self, buf):
        if getattr(self, "volatile", False):
            [val] = self.operands
            align = ", align %d" % self.align if self.align is not None else ""
            buf.append("load volatile {0}, {1} {2}{3}{4}\n".format(
                self.type, val.type, val.get_reference(), align,
                self._stringify_metadata(leading_comma=True)))
        else:
            _ld(self, buf)

    ir.LoadInstr.descr = load_descr
    _st = ir.StoreInstr.descr

    def store_descr(self, buf):
        if getattr(self, "volatile", False):
            val, ptr = self.operands
            align = ", align %d" % self.align if self.align is not None else ""
            buf.append("store volatile {0} {1}, {2} {3}{4}{5}\n".format(
                val.type, val.get_reference(), ptr.type, ptr.get_reference(),
                align, self._stringify_metadata(leading_comma=True)))
        else:
            _st(self, buf)

    ir.StoreInstr.descr = store_descr
    ir.LoadInstr._slifter_volatile_patched = True


_patch_volatile_memops()


class UnsupportedOperatorException(Exception):
    pass


class InvalidTypeException(Exception):
    pass


class NVVMLifter(Lifter):
    def _pre_transform_module(self, module):
        # Cross-function CALL-ABI inlining: merge device-helper Functions
        # (separate top-level `.text` sections reached by `CALL.ABS.NOINC 0x0`
        # + R_CUDA_ABS55_16_34 relocs) into their caller kernels so the
        # intra-function inline path (HandleCall) can lower them. Runs at the
        # module level before per-function transforms split the Functions apart.
        from transform.cross_func_inline import CrossFuncInline
        CrossFuncInline().apply(module)

    def get_transform_passes(self):
        cluster_barriers = ("UCGABAR_ARV", "UCGABAR_WAIT")
        return [
            BuildCFG(),
            # Restore warp-synchronous semantics broken by PBRA branch-splitting
            # of predicated shared-memory ops (cg_warpreduce/graphExecution/pns).
            # Runs on the fresh PBRA structure from BuildCFG.
            WarpReconverge(),
            ResolveBRX(),
            # Resolve register-indirect SASS CALLs via the bank-4 pointer table
            # (sm_90 reaches external device-runtime functions like __assertfail
            # and vprintf through c[0x4]). UNDEF symbols become CALLEXT <name>;
            # in-cubin defined symbols become direct CALL.REL for HandleCall to
            # inline. Must run before HandleCall so the rewritten direct calls
            # participate in its extract/inline machinery, and before SSA so the
            # CALLEXT's ABI register operands (R4:R5 / R6:R7) are versioned.
            ResolveIndirectCall(),
            HandleCall(),
            CFGSimplify(),
            CGPatterns(),
            SRSubstituteReverse(),
            OpModTransform(),
            # Lower predicated defs (`@P INST Rd`) to `INST Rd'; SEL Rd, Rd', Rd_old, P`
            # before SSA, so both the conditional def and its predicate are tracked.
            PredicatedToSel(),
            SSA(),
            DefUseAnalysis(),
            # Collapse the SASS Granlund-Möller magic-divide sequence into a single
            # SDIV pseudo-op; lifted faithfully the chain yields off-by-one quotients
            # for divisors like 1024. Needs SSA + def-use for the reaching-defs walk.
            IntDivRecognize(),
            DefUseAnalysis(),
            RegRemap(),
            DefUseAnalysis(),
            XmadToImad(),
            FPHack(),
            RegRemap(),
            SetZero(),
            DefUseAnalysis(),
            MovEliminate(),
            DefUseAnalysis(),
            DCE(extra_side_effect=cluster_barriers),
            CFGSimplify(),
            DefUseAnalysis(),
            # OperAggregate(),  # Temporarily disabled
            RegRemap(),
            DefUseAnalysis(),
            DCE(extra_side_effect=cluster_barriers),
            CFGSimplify(),
            DefUseAnalysis(),
            TypeAnalysis(),
            # No DefUseAnalysis here: TypeAnalysis only writes op.type_desc
            # tags, never modifies defs/uses or instruction structure, so
            # the previous DefUseAnalysis result is still valid for RegRemap.
            # Saves one ~10s pass per function on logic-rewrite-class kernels.
            RegRemap(),
            # Refresh def-use after the final RegRemap so lift-time consumers
            # (e.g. lift_shfl's active-mask trace) see reaching_defs keyed on
            # the final, renumbered operands.
            DefUseAnalysis(),
        ]

    def setup_module(self, llvm_module):
        llvm_module.triple = "nvptx64-nvidia-cuda"
        llvm_module.data_layout = "e-i64:64-v16:16-v32:32-n16:32:64"
        nvvm_version_node = llvm_module.add_metadata(
            [
                self.ir.Constant(self.ir.IntType(32), 2),
                self.ir.Constant(self.ir.IntType(32), 0),
            ]
        )
        llvm_module.add_named_metadata("nvvmir.version").add(nvvm_version_node)

        # Size @shared_mem to the cubin's ACTUAL largest static __shared__ usage
        # (.nv.shared.<kernel> records the real per-kernel size).
        #
        # We must NOT keep a 32 KB floor: a blanket `.extern .shared .b8
        # shared_mem[32768]` RESERVES 32 KB per block, which at the default 48 KB
        # shared carveout caps occupancy to ~1 block/SM (12%) and causes >100x
        # slowdown on memory-bound atomic loops (the NVVM "timeout cluster":
        # threadfence/filter/atomicPerf/aop/bh/expdist/hungarian/rsc/sc all run
        # >400s lifted). The reference reserves only the real per-kernel .shared
        # (4 B-2 KB). Sizing to max_shared_size() restores occupancy.
        #
        # This cannot under-reserve: every shared-using kernel carries a
        # .nv.shared section, so max_shared_size() >= real usage (tensorT's 47200 B
        # `double tile[5900]` is covered — need > 32768 just works). It cannot
        # regress correctness either: a sized .extern shared CONFLICTS with a
        # dynamic-shared launch, so any dynamic-shared kernel already fails today
        # (segment-reduce: cuLaunchKernel "invalid argument") -> no passing kernel
        # is dynamic-shared. (segment-reduce's dynamic case needs a separate
        # `[0 x i8]` extern, tracked separately.)
        shared_bytes = 32768  # default for the nvdisasm/no-ELF path (fits those)
        elf = getattr(self, "elf", None)
        # max_shared_size() exists on the ELF backend; the nvdisasm single-file
        # path supplies an Asm object with no ELF/.nv.shared section, so guard the
        # call (else AttributeError crashes lift for ALL nvdisasm-path kernels).
        static_shared = 0
        if elf is not None and hasattr(elf, "max_shared_size"):
            need = elf.max_shared_size()
            if need > 0:
                static_shared = need
                shared_bytes = (need + 15) & ~15

        # DYNAMIC-shared kernels (e.g. CUB reduce_by_key in segment-reduce) take
        # their shared size from the launch (`extern __shared__`), NOT from the
        # cubin's static .nv.shared. A SIZED `.extern .shared .b8 shared_mem[N]`
        # CONFLICTS with a dynamic-shared launch arg -> cuLaunchKernel returns
        # "invalid argument". For those kernels @shared_mem must be a ZERO-length
        # `[0 x i8]` (PTX `.extern .shared .b8 name[]`), sized at launch.
        #
        # The verdict was computed on the RAW (pre-transform) module in
        # lift_module, because the discriminator (see _detect_dynamic_shared)
        # needs the UNIFORM-register distinction that RegRemap later erases.
        # We must be CONSERVATIVE: a false "dynamic" verdict on a static kernel
        # would emit `[0 x i8]` and corrupt its real __shared__ storage, so the
        # detector only fires when the full dynamic-shared base dataflow link is
        # established AND the resulting uniform base is used DIRECTLY to address
        # shared memory (see _detect_dynamic_shared for the rationale).
        # The detector covers both classes: static>0 kernels whose dyn-base
        # dataflow link is established, and zero-static sm_90+ kernels whose
        # only shared use is necessarily dynamic (FA3/CUTLASS: mbarrier+TMA
        # windows with no .nv.shared record). It sets _dyn_window_base to the
        # window offset where the dynamic region begins.
        dynamic_shared = getattr(self, "_dynamic_shared_detected", False)

        # In the dynamic-shared case, @shared_mem ([0 x i8], sized at launch)
        # models ONLY the dynamic region. But the SASS addresses shared through
        # the cluster-window base `(CgaCtaId<<24) | static_shared` (the
        # `UMOV imm == static_shared` + `ULEA ...,0x18` idiom), i.e. every
        # shared byte index carries a `+ static_shared` term because on HW the
        # dynamic region begins right after the static .nv.shared region. Our
        # flat dynamic @shared_mem is based at 0, so that `+ static_shared`
        # offset must be SUBTRACTED to rebase the dynamic region to 0 — else
        # accesses run off the end of a small dynamic launch (rsc-cuda launches
        # only sizeof(int) dynamic bytes yet indexes at byte 0x400). See
        # _shared_gep. (segment-reduce launches a large dynamic region so it
        # happened to stay in bounds even un-rebased; the subtraction keeps it
        # correct too.)
        self._shared_dyn_base = (
            getattr(self, "_dyn_window_base", static_shared)
            if dynamic_shared else 0
        )
        if dynamic_shared:
            shared_array_ty = self.ir.ArrayType(self.ir.IntType(8), 0)
        else:
            shared_array_ty = self.ir.ArrayType(self.ir.IntType(8), shared_bytes)
        self.SharedMem = self.ir.GlobalVariable(
            llvm_module, shared_array_ty, "shared_mem", 3
        )
        # Without an explicit alignment, llvmlite emits `.align 1` and ptxas
        # may issue wide (b32/b64/v2.b64) loads/stores whose effective address
        # is not naturally aligned, faulting at launch with
        # CUDA_ERROR_MISALIGNED_ADDRESS. Force 16-byte alignment to match the
        # widest vector load the codegen can emit (v2.b64 = 16 B).
        self.SharedMem.align = 16

        self._local_mem = None
        # Per-thread alloca holding the marshalled kernel-arg region (const
        # bank 0). Created per-function in _lift_function. Replaces storing
        # params into the *global* @const_mem, which made every thread in every
        # block write the SAME global addresses at entry -> L2 same-address
        # write serialization -> ~15-70x slowdown on otherwise-trivial kernels
        # (the FAIL_TIMEOUT cluster: pitch/atomicPerf/aop/...). A per-thread
        # alloca has no cross-thread contention and SROA promotes it straight
        # back to registers, so the generated PTX has ZERO const_mem global
        # ld/st and ZERO local-mem ops. Banks 1-4 (jump tables, c[0x4] device
        # globals) are unchanged — they stay on the global @const_mem.
        self._param_local = None
        self._setup_nv_globals(llvm_module)

    # Opcodes that address shared memory; their first memory operand's base
    # register is the shared base we trace into.
    _SHARED_MEM_OPCODES = (
        "LDS", "STS", "ATOMS", "REDS", "LDSM", "STSM",
    )

    def _detect_dynamic_shared(self, module):
        """Conservative detector for `extern __shared__` (dynamic) shared use.

        MUST be run on the RAW, pre-RegRemap module: it depends on the
        UNIFORM-register (URx) vs normal-register (Rx) distinction that
        RegRemap later collapses into a single `R%d` namespace.

        The verdict is established by a DATAFLOW LINK, not by independently
        spotting the idiom pieces (which ptxas emits for *every* sm_90 shared
        kernel and so cannot discriminate static from dynamic):

          UMOV/UMOV32I URa, <imm>            ; imm == max_shared_size() — the
                                             ;   dyn-shared base offset: the
                                             ;   dynamic region begins right
                                             ;   after the static .nv.shared
          S2UR/S2R     URb, SR_CgaCtaId      ; cluster-window CTA rank
          ULEA         URc, URb, URa, ...    ; URc = (rank<<24)|base  (THE
                                             ;   dynamic-shared base pointer)
          LDS/STS/...  [URc + off]           ; URc used DIRECTLY as a shared
                                             ;   address base operand

        TWO conditions are BOTH required, because each alone is defeated:

        - `imm == static_shared`. The cluster-window base constant ptxas emits
          is the fixed 0x400 for EVERY sm_90 shared kernel (static or dynamic);
          it only equals the kernel's own static .nv.shared size when that size
          is itself 0x400. This excludes static kernels whose shared size
          differs (scan3 0x3410, threadfence 0x410, atomicPerf 0xc00 — all do
          `UMOV 0x400` but 0x400 != their static size, so they stay sized). It
          does NOT, alone, separate scatterAdd (static 0x400) from segment-
          reduce (static 0x400): both pass this test.

        - the dyn-base ULEA result is used DIRECTLY as a UNIFORM shared base
          (`LDS/STS [URc+...]`). A genuine dynamic-shared kernel (CUB
          reduce_by_key in segment-reduce) addresses shared through the uniform
          ULEA result directly (119 such accesses). scatterAdd, despite its
          0x400 static size, routes the ULEA result through a normal-register
          `LEA Rx, Ry, URc` FIRST, so NONE of its shared accesses has the
          uniform base URc directly (0 such) -> it stays sized.

        Only when BOTH hold is the kernel flagged dynamic. Conservative: if the
        link cannot be established, treat as STATIC (sized), never [0].
        """
        self._dyn_window_base = 0
        elf = getattr(self, "elf", None)
        if elf is None or not hasattr(elf, "max_shared_size"):
            return False
        static_shared = elf.max_shared_size()

        if not static_shared or static_shared <= 0:
            # ZERO-STATIC case (FA3 / warp-specialized CUTLASS Hopper
            # kernels): the cubin records NO .nv.shared at all, yet the SASS
            # addresses shared memory (mbarriers, TMA tile buffers). Static
            # `__shared__` always records .nv.shared, so shared use here is
            # necessarily `extern __shared__` (dynamic) — no dataflow link
            # needed. The window offsets these kernels carry are biased by
            # the sm_90 1 KiB reserved SW region (cuobjdump -res-usage shows
            # SHARED:1024 = reserved only), so the rebase base is 0x400.
            # Gated to sm_90+: earlier ISAs have no reserved region /
            # cluster window and their suites were validated sized.
            isa = getattr(module, "isa", "") or ""
            if not isa.startswith(("sm_90", "sm_100", "sm_120")):
                return False
            if self._module_uses_shared(module):
                self._dyn_window_base = 1024
                return True
            return False

        for func in module.functions:
            blocks = func.blocks if func.blocks else [func]
            if not blocks:
                continue
            if self._func_uses_dynamic_shared(blocks, static_shared):
                self._dyn_window_base = static_shared
                return True
        return False

    # Shared-memory users beyond the direct LDS/STS family: mbarrier ops
    # (SYNCS) and TMA tile loads/stores address shared through the same
    # cluster window; LDGSTS's destination is shared. Used only by the
    # zero-static discriminator above (any shared use there => dynamic).
    _SHARED_TOUCH_OPCODES = (
        "LDS", "STS", "ATOMS", "REDS", "LDSM", "STSM",
        "SYNCS", "UTMALDG", "UTMASTG", "LDGSTS",
    )

    def _module_uses_shared(self, module):
        for func in module.functions:
            blocks = func.blocks if func.blocks else [func]
            for bb in blocks:
                insts = getattr(bb, "instructions", None)
                if insts is None:
                    continue
                for inst in insts:
                    opc = inst.opcodes[0] if inst.opcodes else ""
                    if opc in self._SHARED_TOUCH_OPCODES:
                        return True
        return False

    def _func_uses_dynamic_shared(self, blocks, static_shared):
        # (1) uniform-mov of the dyn-base offset: dest URa of `UMOV URa, imm`
        #     where imm == static_shared (the offset where dynamic shared
        #     begins, right after the static .nv.shared region — which on
        #     sm_90 already includes the 1 KiB reserved SW region, so a
        #     zero-static kernel has static_shared == 0x400 and matches the
        #     canonical `UMOV 0x400` window-base directly).
        umov_base_dests = set()
        # (2) CgaCtaId-read dest registers (cluster rank)
        cgactaid_dests = set()
        # rank|base combine defs and shared-base uses, gathered first so the
        # trace is order-independent / SSA-agnostic. ptxas emits the combine
        # as either `ULEA URc, URrank, URbase, ...` or the byte-permute
        # `UPRMT URc, URrank, 0x654, URbase` ((rank<<24)|base — FA3's form).
        combine_insts = []
        # base registers used directly to address shared memory — restricted to
        # UNIFORM registers (the genuine dynamic-shared base is a UR).
        uniform_shared_base_regs = set()

        for bb in blocks:
            insts = getattr(bb, "instructions", None)
            if insts is None:
                continue
            for inst in insts:
                opc = inst.opcodes[0] if inst.opcodes else ""
                if not inst.operands:
                    continue

                if opc in ("UMOV", "UMOV32I", "MOV", "MOV32I"):
                    dst = inst.operands[0]
                    if getattr(dst, "reg", None) and any(
                        op.is_immediate
                        and op.immediate_value == static_shared
                        for op in inst.operands[1:]
                    ):
                        umov_base_dests.add(dst.reg)

                elif opc in ("S2UR", "S2R"):
                    dst = inst.operands[0]
                    if getattr(dst, "reg", None) and any(
                        getattr(op, "is_cluster_ctarank", False)
                        for op in inst.operands
                    ):
                        cgactaid_dests.add(dst.reg)

                elif opc in ("ULEA", "UPRMT"):
                    combine_insts.append(inst)

                # Shared-touching ops: LDS/STS family plus the mbarrier
                # (SYNCS) and TMA (UTMALDG/UTMASTG) ops FA3-class kernels
                # address their dynamic window with.
                if opc in self._SHARED_MEM_OPCODES or \
                        opc in self._SHARED_TOUCH_OPCODES:
                    base = self._shared_base_reg(inst)
                    if base is not None and base.startswith("U"):
                        uniform_shared_base_regs.add(base)

        if not (umov_base_dests and cgactaid_dests and uniform_shared_base_regs):
            return False

        # (3) a combine (ULEA/UPRMT) consuming BOTH the cluster-rank reg AND
        # the dyn-base-offset reg -> its dest is the dynamic-shared base
        # pointer (the (rank<<24)|dyn_off cluster-window encoding).
        dyn_base_dests = set()
        for inst in combine_insts:
            dst = inst.operands[0]
            if not getattr(dst, "reg", None):
                continue
            src_regs = {getattr(op, "reg", None) for op in inst.operands[1:]}
            if (src_regs & cgactaid_dests) and (src_regs & umov_base_dests):
                dyn_base_dests.add(dst.reg)

        if not dyn_base_dests:
            return False

        # (4) the dynamic-shared base pointer is used DIRECTLY as a uniform
        # shared address base. This (combined with imm == static_shared)
        # separates genuine dynamic-shared kernels — which address shared
        # through the uniform ULEA result directly — from static kernels that
        # route the same cluster constant through a normal register first.
        return bool(dyn_base_dests & uniform_shared_base_regs)

    @staticmethod
    def _shared_base_reg(inst):
        """Return the base register name of a shared-memory instruction's
        first memory-address operand, or None."""
        for op in inst.operands:
            if not getattr(op, "is_mem_addr", False):
                continue
            if op.is_reg_group and op.reg_group and op.reg_group[0].reg:
                return op.reg_group[0].reg
            if getattr(op, "reg", None):
                return op.reg
            return None
        return None

    def _setup_nv_globals(self, llvm_module):
        """Materialize the cubin's __device__ globals in global memory.

        `.nv.global.init` (PROGBITS) and `.nv.global` (NOBITS) hold __device__
        globals — perlin's gradient/permutation tables, the CUDA math runtime's
        `__cudart_i2opi_d`/`__cudart_sin_cos_coeffs` trig-reduction tables
        (fft/gabor/romberg), etc. SASS reaches them through const bank 4: the
        loader writes each global's address into a c[0x4] slot (R_CUDA_64), and
        the body does `LDC.64 Rd, c[0x4][slot]` then `LDG.E [Rd]`. The lifted
        PTX is a standalone module, so it must define its OWN copies here;
        otherwise c[0x4] reads back 0 and every table base null-derefs.
        Keyed by in_init_section (True = .nv.global.init). See
        _nv_global_const4_addr for the c[0x4] -> address substitution.
        """
        self._nv_globals = {}
        elf = getattr(self, "elf", None)
        if elf is None:
            return
        i8 = self.ir.IntType(8)
        init_bytes = getattr(elf, "nv_global_init", b"") or b""
        if init_bytes:
            arr_ty = self.ir.ArrayType(i8, len(init_bytes))
            g = self.ir.GlobalVariable(
                llvm_module, arr_ty, "__slifter_nv_global_init", 1)
            g.initializer = self.ir.Constant(arr_ty, bytearray(init_bytes))
            g.align = 16
            self._nv_globals[True] = g
        bss_size = getattr(elf, "nv_global_size", 0) or 0
        if bss_size:
            arr_ty = self.ir.ArrayType(i8, bss_size)
            g = self.ir.GlobalVariable(
                llvm_module, arr_ty, "__slifter_nv_global", 1)
            g.initializer = self.ir.Constant(arr_ty, None)
            g.align = 16
            self._nv_globals[False] = g

    def _nv_global_const4_addr(self, builder, byte_offset, target_ty, name):
        """c[0x4][slot] holds a loader-relocated device-global base (R_CUDA_64).
        Return the address of our emitted addrspace(1) device global instead of
        the section's pre-relocation zero. The 8-byte pointer slot is normally
        read whole (LDC.64 -> i64); handle a lo/hi i32 half as a fallback."""
        elf = getattr(self, "elf", None)
        if elf is None:
            return None
        relocs = getattr(elf, "const4_relocs", None) or {}
        syms = getattr(elf, "nv_global_syms", None) or {}
        is_hi = False
        rec = relocs.get(byte_offset)
        if rec is None and (byte_offset - 4) in relocs:
            rec = relocs.get(byte_offset - 4)
            is_hi = True
        if rec is None:
            return None
        symname, _val, is_func = rec
        if is_func:
            # Device-function pointers (indirect CALL table) are resolved
            # elsewhere, not here.
            return None
        sym = syms.get(symname)
        if sym is None:
            return None
        in_init, sym_off, _size = sym
        g = self._nv_globals.get(in_init)
        if g is None:
            return None
        i64 = self.ir.IntType(64)
        p = builder.gep(
            g, [self.ir.Constant(i64, 0), self.ir.Constant(i64, sym_off)],
            f"{name}_nvg")
        addr = builder.ptrtoint(p, i64, f"{name}_nvg_addr")
        width = target_ty.width if isinstance(target_ty, self.ir.IntType) else 64
        if width >= 64:
            return addr
        if is_hi:
            addr = builder.lshr(addr, self.ir.Constant(i64, 32), f"{name}_hi")
        return builder.trunc(addr, target_ty, f"{name}_half")

    def _nv_global_addr(self, builder, symname, name="nvg"):
        """i64 address of a materialised __device__-global symbol, or None.
        Mirrors `_nv_global_const4_addr` but keyed by symbol name (used for the
        ABS32-immediate path where the global base is patched into MOV/UMOV
        zero-immediates rather than a c[0x4] slot)."""
        syms = getattr(self.elf, "nv_global_syms", None) or {}
        rec = syms.get(symname)
        if rec is None:
            return None
        in_init, sym_off, _size = rec
        g = self._nv_globals.get(in_init)
        if g is None:
            return None
        i64 = self.ir.IntType(64)
        p = builder.gep(
            g, [self.ir.Constant(i64, 0), self.ir.Constant(i64, sym_off)],
            f"{name}_nvg")
        return builder.ptrtoint(p, i64, f"{name}_nvg_addr")

    def _abs32_addr_half(self, Inst, IRBuilder):
        """R_CUDA_ABS32_LO/HI_32: this MOV/UMOV's zero immediate is patched by
        the loader with one 32-bit half of a __device__-global's address. The
        SASS materialises a coefficient-table base as a `MOV Rlo, 0x0` /
        `MOV Rhi, 0x0` pair the loader relocates (fresnel's Chebyshev/
        trig-reduction tables — `__cudart_i2opi_d`, Chebyshev_Expansion_*::c[]).
        The lifted module is standalone, so resolve to OUR materialised global
        instead of leaving the immediate 0 (-> base 0 -> tiny OOB read, e.g.
        addr 0xa0). `Inst.id` is the SASS text offset = the relocation key;
        cross_func_inline rebases callee offsets and merges their relmaps into
        the kernel's, so the lookup works after inlining too."""
        if not getattr(self, "_nv_globals", None):
            return None
        relmap = (getattr(self.elf, "rel_text_abs32", None) or {}).get(
            self.curr_function.name)
        if not relmap:
            return None
        # After cross-function inlining the instruction id is a compound
        # `<kernel_off>_<callee_off>_<deeper_off>` chain (each `_` level a deeper
        # inline). The merged relmap is keyed by the REBASED offset of the level
        # the MOV physically came from — the LAST component (deepest inline).
        # A non-inlined MOV has a bare `<off>` id, so trying all `_`-components
        # (last first) covers both. The `@` suffix is the SSA version tag.
        ent = None
        for part in reversed(str(Inst.id).split("_")):
            try:
                off = int(part.split("@")[0], 16) & ~0xF
            except (ValueError, AttributeError):
                continue
            ent = relmap.get(off)
            if ent is not None:
                break
        if ent is None:
            return None
        symname, is_hi = ent
        addr = self._nv_global_addr(IRBuilder, symname, "nvg")
        if addr is None:
            return None
        i64 = self.ir.IntType(64)
        if is_hi:
            addr = IRBuilder.lshr(addr, self.ir.Constant(i64, 32), "nvg_hi")
        return IRBuilder.trunc(addr, self.ir.IntType(32), "nvg_half")

    @lift_for("MOV", "MOV64", "UMOV", "R2UR")
    def lift_mov(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        half = self._abs32_addr_half(Inst, IRBuilder)
        if half is not None:
            self._set_val(IRBuilder, IRRegs, Inst.get_defs()[0], half)
            return
        return super().lift_mov(Inst, IRBuilder, IRRegs, block_map=block_map,
                                _get_val=_get_val, **kwargs)

    @property
    def LocalMem(self):
        return self._local_mem

    def _load_from_const_mem(self, builder, bank, byte_offset, target_ty, name=""):
        # sm_90 driver slot c[0x0][0x198]: the GENERIC ADDRESS of the kernel's
        # grid-constant param buffer (nvcc's cvta.param materialization; TMA
        # descriptor chains are rooted here: desc = c[0x198] + 0x30 + off).
        # The original module's slot value is meaningless under our launch —
        # substitute cvta.param of OUR param_0 minus the fixed 0x30 ABI bias so
        # downstream arithmetic lands on our own (identical-layout) params.
        fn_name = builder.function.name
        param_bias = 0x30 if byte_offset == 0x198 else None
        if bank == 0 and byte_offset == 0x36c and getattr(self.elf, "blackwell_param_abi", False) and target_ty == self.ir.IntType(32):
            # Verified with mov.pred %is_explicit_cluster on sm_100a/sm_120.
            read_cluster = self.ir.InlineAsm(self.ir.FunctionType(target_ty, []),
                "{ .reg .pred p; mov.pred p, %is_explicit_cluster; selp.u32 $0, 1, 0, p; }",
                "=r", side_effect=False)
            return builder.call(read_cluster, [], name or "explicit_cluster")
        # Blackwell cvta.param materializes the address of param_0 directly.
        if byte_offset == 0x348 and getattr(self.elf, "blackwell_param_abi", False):
            param_bias = 0
        # The older NVJET Hopper cubins load the bank-0 base from 0xc0.
        # Their TMA offsets are absolute bank offsets (e.g. 0x2c0 for
        # the descriptor 176 bytes into the argument at 0x210).
        if byte_offset == 0xc0 and fn_name.startswith("nvjet_sm90_") and self.arg_info:
            param_bias = min(self.arg_info)
        if bank == 0 and param_bias is not None and \
                isinstance(target_ty, self.ir.IntType) and target_ty.width == 64:
            i64 = self.ir.IntType(64)
            asm = self.ir.InlineAsm(
                self.ir.FunctionType(i64, []),
                "{ .reg .b64 t; mov.b64 t, %s_param_0; cvta.param.u64 $0, t; }"
                % fn_name,
                "=l", side_effect=False)
            base = builder.call(asm, [], name or "gc_param_base")
            return builder.sub(base, self.ir.Constant(i64, param_bias),
                               (name or "gc_param_base") + "_biased")
        # Bank 4 holds loader-relocated __device__-global base addresses
        # (R_CUDA_64). Resolve to our materialized addrspace(1) globals.
        if bank == 4:
            addr = self._nv_global_const4_addr(
                builder, byte_offset, target_ty, name or "c4")
            if addr is not None:
                return addr
        # Bank-0 marshalled kernel args live in the per-thread alloca, not the
        # contended global @const_mem. Lift-time constants (frame size at 0x28,
        # ELF const data) still resolve first via the base helper.
        if bank == 0 and self._param_local is not None:
            cv = self._load_const_from_elf_const_data(bank, byte_offset, target_ty)
            if cv is not None:
                return cv
            if self._should_load_const_from_memory(0, byte_offset):
                ptr = builder.gep(
                    self._param_local,
                    [self.ir.Constant(self.ir.IntType(64), 0),
                     self.ir.Constant(self.ir.IntType(64), byte_offset)],
                    f"{name}_ptr" if name else "pl_ptr",
                )
                ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty, 0))
                return builder.load(ptr, name if name else "pl_load")
        return super()._load_from_const_mem(builder, bank, byte_offset, target_ty, name)

    def _load_from_const_mem_dynamic(self, builder, bank, byte_offset_val, target_ty, name=""):
        # Dynamic (register-indexed) bank-0 reads index into the marshalled
        # arg region -> the per-thread alloca.
        if bank == 0 and self._param_local is not None:
            if byte_offset_val.type != self.ir.IntType(64):
                if byte_offset_val.type.width < 64:
                    byte_offset_val = builder.zext(byte_offset_val, self.ir.IntType(64), f"{name}_off_zext")
                else:
                    byte_offset_val = builder.trunc(byte_offset_val, self.ir.IntType(64), f"{name}_off_trunc")
            ptr = builder.gep(
                self._param_local,
                [self.ir.Constant(self.ir.IntType(64), 0), byte_offset_val],
                f"{name}_ptr" if name else "pl_dyn_ptr",
            )
            ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty, 0))
            return builder.load(ptr, name if name else "pl_dyn_load")
        return super()._load_from_const_mem_dynamic(builder, bank, byte_offset_val, target_ty, name)

    def _get_ir_function_type(self, func):
        if not (self.arg_map and func.name in self.arg_map):
            return self.ir.FunctionType(self.ir.VoidType(), [], False)

        # Drive param list from cubin's KPARAM_INFO (arg_map), not from the
        # operand walk. SASS may legitimately not reference every parameter
        # (e.g. an unused float scalar or a "tag" param the compiler
        # constant-folded out). If we drop unused params, ptxas re-lays
        # the kernel's .param block at compacted offsets and cuLaunchKernel
        # then writes the host's args[] into wrong cubin slots — every
        # arg after the first dropped one is shifted by its size.
        param_types = []
        for entry in self.arg_map[func.name]:
            if entry is None:
                continue
            _arg_offset, arg_size = entry
            param_types.append(self._arg_ir_type(arg_size))

        return self.ir.FunctionType(self.ir.VoidType(), param_types, False)

    def _arg_ir_type(self, arg_size):
        # NVPTX's LowerArguments / getCopyFromParts segfaults on integer
        # arg widths it can't lower cleanly: anything non-power-of-2
        # (i24/i48/i96/i384/...) and anything wider than i128. CUTLASS
        # reference kernels take e.g. a 12-byte GemmCoord as i96; a single
        # ~368-byte Kernel::Params would be i2944. Both crash. Crashes
        # repro on LLVM 14 *and* LLVM 21 — upstream NVPTX bug.
        #
        # Byte arrays match the cubin ABI (.param .b8 name[N]) exactly
        # and are lowered iteratively without recursion. Stay on packed
        # integers only for the natural byte widths (1, 2, 4, 8, 16) so
        # existing tests that pass i32/i64/i128 args continue to work.
        if arg_size in (1, 2, 4, 8, 16):
            return self.ir.IntType(arg_size * 8)
        # 64-byte-multiple aggregates: use [N/64 x <8 x i64>] so NVPTX emits
        # `.param .align 64 .b8 name[N]` (same size/ABI — the driver still
        # copies N raw bytes — but the param VA is 64-aligned). Required for
        # __grid_constant__ CUtensorMap-carrying Params (FA3/CUTLASS): TMA
        # reads the tensormap through cvta.param-rooted generic addresses and
        # faults MISALIGNED unless the map (64-aligned WITHIN the struct) is
        # 64-aligned in memory. An [N x i8] param is only `.align 1`, and
        # LLVM cannot express alignment on a by-value aggregate arg (align
        # attr is pointer-only; byval lowers to an ABI-changing .ptr param).
        if arg_size % 64 == 0:
            return self.ir.ArrayType(
                self.ir.VectorType(self.ir.IntType(64), 8), arg_size // 64)
        return self.ir.ArrayType(self.ir.IntType(8), arg_size)

    def _should_load_const_from_memory(self, bank, byte_offset):
        if bank == 0:
            for arg_off, (_, arg_size) in self.arg_info.items():
                if arg_off <= byte_offset < arg_off + arg_size:
                    return True
        return False

    def _get_intrinsic(self, name):
        if name in self.intrinsics:
            return self.intrinsics[name]
        
        i1 = self.ir.IntType(1)
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        f32 = self.ir.FloatType()
        f64 = self.ir.DoubleType()

        func = None
        if name == "abs":
            func_ty = self.ir.FunctionType(i32, [i32, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.abs.i32")
        elif name == "fabs":
            func_ty = self.ir.FunctionType(f32, [f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fabs.f")
        elif name == "fabs_f64":
            func_ty = self.ir.FunctionType(f64, [f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fabs.d")
        elif name == "hmma884f32":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([f32] * 8)
            func_ty = ir.FunctionType(
                output_ty, [v2f16] * 4 + [f32] * 8, False
            )
            func = ir.Function(
                self.llvm_module,
                func_ty,
                "llvm.nvvm.mma.m8n8k4.col.row.f32.f32",
            )
        elif name == "hmma884f16":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([v2f16] * 4)
            func_ty = ir.FunctionType(
                output_ty, [v2f16] * 4 + [v2f16] * 4, False
            )
            func = ir.Function(
                self.llvm_module,
                func_ty,
                "llvm.nvvm.mma.m8n8k4.col.row.f16.f16",
            )
        elif name == "hmma1688f32":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(
                output_ty, [v2f16, v2f16, v2f16, f32, f32, f32, f32], False
            )
            func = ir.Function(
                self.llvm_module,
                func_ty,
                "llvm.nvvm.mma.m16n8k8.row.col.f32.f32",
            )
        elif name == "hmma1688f32_tf32":
            # HMMA.1688.F32.TF32 -> llvm.nvvm.mma.m16n8k8.row.col.tf32.
            # Base handler passes A0-3, B0-1 as f32 (one tf32 per full reg)
            # and C0-3 f32; the intrinsic wants the tf32 operands as i32 bit
            # patterns. Was MISSING on nvvm — 14_ampere_tf32_tensorop_gemm's
            # kernel hit the unknown-intrinsic path and produced wrong output.
            i32 = ir.IntType(32)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(
                output_ty, [i32] * 6 + [f32] * 4, False
            )
            func = ir.Function(
                self.llvm_module,
                func_ty,
                "llvm.nvvm.mma.m16n8k8.row.col.tf32",
            )
            # A/B arrive as f32 (tf32-per-full-reg); run_intrinsic's generic
            # arg-cast bitcasts them to the intrinsic's i32 params.
        elif name == "hmma1688f16":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([v2f16, v2f16])
            func_ty = ir.FunctionType(
                output_ty, [v2f16, v2f16, v2f16, v2f16, v2f16], False
            )
            func = ir.Function(
                self.llvm_module,
                func_ty,
                "llvm.nvvm.mma.m16n8k8.row.col.f16.f16",
            )
        elif name == "hmma16816f32":
            # SM80+ m16n8k16 f16-input, f32-accum MMA (HMMA.16816.F32) — the
            # shape almost all modern wmma/CUTLASS kernels use. Was MISSING
            # (only the k8 hmma1688 variants existed) so HMMA.16816 hit an
            # undefined intrinsic -> 0 result (tensor_simple, tensor_*, many
            # CUTLASS gemms). A:4 + B:2 = 6 v2f16, then 4 f32 accumulators.
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([f32] * 4)
            func_ty = ir.FunctionType(
                output_ty, [v2f16] * 6 + [f32] * 4, False
            )
            func = ir.Function(
                self.llvm_module, func_ty,
                "llvm.nvvm.mma.m16n8k16.row.col.f32.f32",
            )
        elif name == "hmma16816f32_bf16":
            # NVVM uses four packed i32 A fragments, two packed i32 B
            # fragments and four f32 accumulators for the bf16 form.
            func_ty = ir.FunctionType(ir.LiteralStructType([f32] * 4),
                                      [ir.IntType(32)] * 6 + [f32] * 4)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.mma.m16n8k16.row.col.bf16")
        elif name == "hmma16816f16":
            # m16n8k16 f16-input, f16-accum MMA. A:4 + B:2 + C:2 = 8 v2f16.
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            output_ty = ir.LiteralStructType([v2f16, v2f16])
            func_ty = ir.FunctionType(output_ty, [v2f16] * 8, False)
            func = ir.Function(
                self.llvm_module, func_ty,
                "llvm.nvvm.mma.m16n8k16.row.col.f16.f16",
            )
        elif name == "dmma884f64":
            # SM80+ m8n8k4 f64 tensor-core MMA (DMMA.8x8x4). The base
            # lifter.lift_dmma emits the x86 warp-gather wrapper
            # (dmma_884_f64), a host helper that becomes an unresolved
            # extern in PTX. On GPU the tensor core does the warp-collective
            # natively. A:1 + B:1 + C:2 = 4 f64 -> {f64, f64}. PTX f64 MMA
            # only supports the .row.col layout.
            output_ty = ir.LiteralStructType([f64, f64])
            func_ty = ir.FunctionType(output_ty, [f64] * 4, False)
            func = ir.Function(
                self.llvm_module, func_ty,
                "llvm.nvvm.mma.m8n8k4.row.col.f64",
            )
        elif name == "dmma16816f64":
            # SM90 m16n8k16 f64 MMA (DMMA.16x8x16, dgemm mainloop).
            # A:8 + B:4 + C:4 = 16 f64 -> {f64 x 4}.
            output_ty = ir.LiteralStructType([f64] * 4)
            func_ty = ir.FunctionType(output_ty, [f64] * 16, False)
            func = ir.Function(
                self.llvm_module, func_ty,
                "llvm.nvvm.mma.m16n8k16.row.col.f64",
            )
        elif name.startswith("imma_"):
            # imma_m8n8k16_s8 -> llvm.nvvm.mma.m8n8k16.row.col.s8
            # imma_m8n8k16_satfinite_s8 -> llvm.nvvm.mma.m8n8k16.row.col.satfinite.s8
            suffix = name[len("imma_"):]
            geom_end = suffix.index("_", suffix.index("k"))
            geom = suffix[:geom_end]
            rest = suffix[geom_end:].replace("_", ".")
            llvm_name = f"llvm.nvvm.mma.{geom}.row.col{rest}"
            # Fragment counts derive from the geometry: int8/uint8 MMA packs 4
            # s8 per i32 over 32 lanes, so A=M*K/128, B=N*K/128 i32 input regs
            # and the s32 accumulator C/D = M*N/32 i32 regs. (m8n8k16 -> 1+1+2.)
            # Hardcoding 4-in/2-out was correct only for the 8816 turing shape.
            gm = re.match(r"m(\d+)n(\d+)k(\d+)", geom)
            M, N, K = int(gm.group(1)), int(gm.group(2)), int(gm.group(3))
            a_regs, b_regs, c_regs = M * K // 128, N * K // 128, M * N // 32
            output_ty = ir.LiteralStructType([i32] * c_regs)
            func_ty = ir.FunctionType(output_ty, [i32] * (a_regs + b_regs + c_regs), False)
            func = ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name in ("membar_cta", "membar_gl", "membar_sys"):
            scope = name[len("membar_"):]
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
            func = self.ir.Function(self.llvm_module, func_ty,
                                    f"llvm.nvvm.membar.{scope}")
        elif name == "hadd2":
            func_ty = ir.FunctionType(i32, [i32, i32], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.f16x2.add")
        elif name == "hfma2":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fma.rn.f16x2")
        elif name == "hfma2_ftz":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fma.rn.ftz.f16x2")
        elif name == "hmin2":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fmin.f16x2")
        elif name == "hmin2_ftz":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fmin.ftz.f16x2")
        elif name == "hmax2":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fmax.f16x2")
        elif name == "hmax2_ftz":
            v2f16 = ir.VectorType(ir.HalfType(), 2)
            func_ty = ir.FunctionType(v2f16, [v2f16, v2f16], False)
            func = ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fmax.ftz.f16x2")
        elif name == "barrier":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.barrier0")
        elif name == "bar_warp_sync":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.bar.warp.sync")
        elif name == "barrier0_and":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.barrier0.and")
        elif name == "barrier0_or":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.barrier0.or")
        elif name == "barrier0_popc":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.barrier0.popc")
        elif name == "activemask":
            # PTX `activemask.b32` — ballot of the currently-active lanes.
            # NOT a %sreg read: llvm.nvvm.read.ptx.sreg.activemask does not
            # exist (llc lowers the unknown name to an unresolvable extern
            # call, which ptxas rejects).
            func_ty = self.ir.FunctionType(i32, [], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.activemask")
        elif name.startswith("vote_"):
            vote_map = {
                "vote_any": ("llvm.nvvm.vote.any.sync", i1, [i32, i1]),
                "vote_all": ("llvm.nvvm.vote.all.sync", i1, [i32, i1]),
                "vote_uni": ("llvm.nvvm.vote.uni.sync", i1, [i32, i1]),
                "vote_ballot": ("llvm.nvvm.vote.ballot.sync", i32, [i32, i1]),
            }
            if name in vote_map:
                llvm_name, ret, arg_tys = vote_map[name]
                func_ty = self.ir.FunctionType(ret, arg_tys, False)
                func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name.startswith("shfl_"):
            parts = name.split("_")
            mode, dtype_name = parts[1], parts[2]
            dtype_map = {"i32": i32, "i64": i64, "f32": f32, "f64": f64}
            dtype = dtype_map.get(dtype_name)
            if dtype:
                llvm_name = f"llvm.nvvm.shfl.sync.{mode}.{dtype_name}"
                func_ty = self.ir.FunctionType(dtype, [i32, dtype, i32, i32], False)
                func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name.startswith("match_any_"):
            dtype_name = name.split("_")[-1]
            dtype = {"i32": i32, "i64": i64}.get(dtype_name)
            if dtype:
                llvm_name = f"llvm.nvvm.match.any.sync.{dtype_name}"
                # match.any.sync returns the 32-bit lane MASK regardless of the
                # matched value's width. LLVM 21's IntrinsicsNVVM.td declares
                # both .i32 and .i64 variants returning i32 (the mask); the old
                # i64-return signature (LLVM 14 era) makes the verifier reject
                # "Intrinsic has incorrect return type" (atomicAggregate). Value
                # arg is `dtype` (i32/i64); return is always i32.
                func_ty = self.ir.FunctionType(i32, [i32, dtype], False)
                func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name.startswith("redux_sync_"):
            # Hardware warp-reduce (sm_80+): redux.sync.<op>.<type> d, a, mask.
            # op in {add,min,umin,max,umax,and,or,xor}; (src, membermask)->i32.
            op = name[len("redux_sync_"):]
            llvm_name = f"llvm.nvvm.redux.sync.{op}"
            func_ty = self.ir.FunctionType(i32, [i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name == "brev32":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.brev32")
        elif name == "brev64":
            func_ty = self.ir.FunctionType(i64, [i64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.brev64")
        elif name == "prmt":
            func_ty = self.ir.FunctionType(i32, [i32, i32, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.prmt")
        elif name == "ctlz_i32":
            func_ty = self.ir.FunctionType(i32, [i32, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctlz.i32")
        elif name == "ctlz_i64":
            func_ty = self.ir.FunctionType(i32, [i64, i1], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctlz.i64")
        elif name == "ctpop_i32":
            func_ty = self.ir.FunctionType(i32, [i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.ctpop.i32")
        elif name == "atom_add_f16x2":
            ptr_ty = self.ir.PointerType(i32, 1)
            func_ty = self.ir.FunctionType(i32, [ptr_ty, i32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "atom_add_f16x2")
        elif name.startswith("isspacep_"):
            space_name = name.split("_")[1]  # const, global, local, shared
            llvm_name = f"llvm.nvvm.isspacep.{space_name}"
            # LLVM 14's IntrinsicsNVVM.td declares isspacep.* as taking
            # `llvm_ptr_ty` (which lowers to i8* in LLVM 14, opaque ptr in
            # 15+). Declaring i64 fails the verifier.
            ptr_ty = self.ir.PointerType(self.ir.IntType(8))
            func_ty = self.ir.FunctionType(i1, [ptr_ty], False)
            func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name.startswith("sreg_"):
            sreg_name = name[5:]
            llvm_name = f"llvm.nvvm.read.ptx.sreg.{sreg_name}"
            # clock64 and the bare globaltimer return i64; everything else
            # (tid.*, ntid.*, clock, globaltimer.lo/hi, lanemask.*, ...) is i32.
            ret_ty = i64 if sreg_name in ("clock64", "globaltimer") else i32
            func_ty = self.ir.FunctionType(ret_ty, [])
            func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        elif name == "fma_f64":
            # SASS DFMA is a fused multiply-add (single rounding); llvm.fma.f64
            # preserves that, vs fmul+fadd which would round twice.
            func_ty = self.ir.FunctionType(f64, [f64, f64, f64], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.fma.f64")
        elif name == "fma_rp_f32":
            # FFMA.RP — round toward +inf.
            func_ty = self.ir.FunctionType(f32, [f32, f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fma.rp.f")
        elif name == "fma_rm_f32":
            # FFMA.RM — round toward -inf.
            func_ty = self.ir.FunctionType(f32, [f32, f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fma.rm.f")
        elif name == "fma_rz_f32":
            # FFMA.RZ — round toward zero.
            func_ty = self.ir.FunctionType(f32, [f32, f32, f32], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.nvvm.fma.rz.f")
        elif name == "trap":
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            func = self.ir.Function(self.llvm_module, func_ty, "llvm.trap")
        else:
            mufu_map = {
                "rcp_f32": "llvm.nvvm.rcp.approx.ftz.f",
                "rsqrt_f32": "llvm.nvvm.rsqrt.approx.ftz.f",
                "sqrt_f32": "llvm.sqrt.f32",
                "sin_f32": "llvm.nvvm.sin.approx.ftz.f",
                "cos_f32": "llvm.nvvm.cos.approx.ftz.f",
                "exp2_f32": "llvm.nvvm.ex2.approx.ftz.f",
                "log2_f32": "llvm.nvvm.lg2.approx.ftz.f",
            }
            rounding_map = {
                "floor_f32": ("llvm.nvvm.floor.f", f32),
                "floor_f64": ("llvm.nvvm.floor.d", f64),
                "ceil_f32": ("llvm.nvvm.ceil.f", f32),
                "ceil_f64": ("llvm.nvvm.ceil.d", f64),
                "trunc_f32": ("llvm.nvvm.trunc.f", f32),
                "trunc_f64": ("llvm.nvvm.trunc.d", f64),
                "rint_f32": ("llvm.nvvm.round.f", f32),
                "rint_f64": ("llvm.nvvm.round.d", f64),
            }
            if name in rounding_map:
                llvm_name, dtype = rounding_map[name]
                func_ty = self.ir.FunctionType(dtype, [dtype], False)
                func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
            elif name in mufu_map:
                llvm_name = mufu_map[name]
                func_ty = self.ir.FunctionType(f32, [f32], False)
                func = self.ir.Function(self.llvm_module, func_ty, llvm_name)
        
        if func is None:
            raise UnsupportedInstructionException(f"Unknown intrinsic {name}")
        
        self.intrinsics[name] = func
        return func

    def run_intrinsic(self, name, builder, args, call_name=""):
        if name == "fchk":
            return ir.Constant(ir.IntType(1), 0)

        # MUFU.RCP64H / RSQ64H: emit raw PTX rcp/rsqrt.approx.f64 rather than the
        # llvm.nvvm.*.approx.d intrinsics, which PTXAS lowers with an extra
        # Newton-Raphson refinement step the original SASS does not contain.
        if name == "rsqrt_f64" and len(args) == 1:
            f64 = self.ir.DoubleType()
            asm_ty = self.ir.FunctionType(f64, [f64])
            inline_asm = self.ir.InlineAsm(
                asm_ty,
                "rsqrt.approx.ftz.f64 $0, $1;",
                "=d,d",
            )
            return builder.call(inline_asm, list(args), call_name or "rsqrt_f64_asm")
        if name == "rcp_f64" and len(args) == 1:
            f64 = self.ir.DoubleType()
            asm_ty = self.ir.FunctionType(f64, [f64])
            inline_asm = self.ir.InlineAsm(
                asm_ty,
                "rcp.approx.ftz.f64 $0, $1;",
                "=d,d",
            )
            return builder.call(inline_asm, list(args), call_name or "rcp_f64_asm")

        # Named/partial barriers (BAR.SYNC/BAR.ARV <id>, <count>): emit the real
        # PTX `bar.sync`/`bar.arrive` over the named barrier — preserves
        # warpgroup-partial synchronization on the GPU path.
        if name in ("named_barrier_sync", "named_barrier_arrive") and len(args) == 2:
            void = self.ir.VoidType()
            i32 = self.ir.IntType(32)
            asm = "bar.sync $0, $1;" if name == "named_barrier_sync" else "bar.arrive $0, $1;"
            asm_ty = self.ir.FunctionType(void, [i32, i32])
            inline_asm = self.ir.InlineAsm(asm_ty, asm, "r,r,~{memory}", side_effect=True)
            return builder.call(inline_asm, list(args), call_name or "nbar_asm")

        func = self._get_intrinsic(name)
        
        
        if name == "abs":
            # llvm.abs.i32 needs (value, is_int_min_poison)
            args = list(args) + [ir.Constant(ir.IntType(1), 0)]
        elif name in ("ctlz_i32", "ctlz_i64"):
            # llvm.ctlz needs is_zero_undef
            args = list(args) + [ir.Constant(ir.IntType(1), 0)]
            
        # match args type to function parameter types
        func_param_tys = [arg for arg in func.function_type.args]
        casted_args = []
        for arg, param_ty in zip(args, func_param_tys):
            if arg.type != param_ty:
                arg = self._insert_type_cast_if_needed(builder, arg, param_ty, f"{call_name}_arg_cast")
            casted_args.append(arg)

        if len(args) != len(func_param_tys):
            raise UnsupportedInstructionException(
                f"intrinsic {name} arg count mismatch: got {len(args)} expected {len(func_param_tys)}"
            )
        
        return builder.call(func, casted_args, call_name)

    def _emit_lane_id_i32(self, builder):
        # Backend hook used by shared lifters (e.g. lift_shfl's destination
        # predicate): the physical lane id within the warp.
        return self.run_intrinsic("sreg_laneid", builder, [], "laneid")

    def _int_to_float_directed(self, IRBuilder, ival, float_ty, is_unsigned, rnd):
        # I2F.RP/.RM (int -> float, round toward +/-Inf). NVPTX has the native
        # directed conversion `cvt.rp/.rm.<ftype>.<itype>` and rounds exactly as
        # the SASS does, so emit it directly. The shared base helper emulates
        # directed rounding with an i64->i128 round-trip (`fptosi double to
        # i128`) that the NVPTX backend cannot select ("unsupported library call
        # operation" -> intrinsics-cast/resize/bitpacking FAIL_LIFT).
        rmode = {"RP":"rp", "RM":"rm", "RZ":"rz"}[rnd]
        w = ival.type.width
        if w not in (32, 64):
            ival = (IRBuilder.zext(ival, self.ir.IntType(64), "i2f_dir_ext")
                    if is_unsigned else IRBuilder.sext(ival, self.ir.IntType(64), "i2f_dir_ext"))
            w = 64
        itype = ("u" if is_unsigned else "s") + str(w)
        in_con = "r" if w == 32 else "l"
        if isinstance(float_ty, self.ir.DoubleType):
            ftype, out_con = "f64", "d"
        elif isinstance(float_ty, self.ir.HalfType):
            ftype, out_con = "f16", "h"
        else:
            ftype, out_con = "f32", "f"
        asm_ty = self.ir.FunctionType(float_ty, [ival.type])
        asm = self.ir.InlineAsm(asm_ty, f"cvt.{rmode}.{ftype}.{itype} $0, $1;",
                                f"={out_con},{in_con}")
        return IRBuilder.call(asm, [ival], "i2f_directed")

    def _store_arg_to_const_mem(self, builder, arg_val, byte_offset, arg_size):
        # Marshal the arg into the per-thread param alloca (bank 0), not the
        # contended global @const_mem. See the _param_local docstring.
        ptr = builder.gep(
            self._param_local,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), byte_offset),
            ],
            f"arg_ptr_{byte_offset}",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(arg_val.type, 0))
        st = builder.store(arg_val, ptr)
        # const_mem is 16-byte aligned, so the slot at `byte_offset` is only
        # gcd(16, byte_offset)-aligned. An 8-byte arg (e.g. a packed NppiSize)
        # at a 4-aligned param offset (0x34, 0x5c, …) would otherwise lower to a
        # natural-alignment st.global.b64 and fault with MISALIGNED_ADDRESS.
        # Mark the true alignment so NVPTX emits a store that respects it.
        st.align = byte_offset & -byte_offset if byte_offset else 16

    def _lift_function(self, func, llvm_module):
        self._setup_const_mem(llvm_module, func.name)

        self.arg_info = {}
        if self.arg_map and func.name in self.arg_map:
            for entry in self.arg_map[func.name]:
                if entry:
                    arg_offset, arg_size = entry
                    self.arg_info[arg_offset] = (arg_offset, arg_size)

        # Param list comes from arg_map, not the operand walk — see
        # _get_ir_function_type for the rationale (unused params silently
        # corrupt cubin layout otherwise).
        ordered_args = []  # list of (offset, size)
        if self.arg_map and func.name in self.arg_map:
            for entry in self.arg_map[func.name]:
                if entry is None:
                    continue
                ordered_args.append(entry)

        param_types = []
        for arg_offset, arg_size in ordered_args:
            # Use byte-exact arg widths so the function definition's
            # signature matches the forward declaration produced by
            # `_get_ir_function_type` (which uses raw `IntType(arg_size * 8)`).
            # The previous power-of-two rounding below caused
            # `Function type mismatch: void(i384) vs void(i512)` for any
            # arg whose natural width wasn't already 8/16/32/64/128
            # (e.g. 48-byte `InvDctParams` -> i384). Cubin metadata
            # (EIATTR_KPARAM_INFO) reports byte-exact sizes, and
            # downstream wrappers cast `args[0]` to `iN*` and load the
            # full blob in one shot — so the type has to match the
            # actual data width, not a rounded-up one.
            #
            # bits = arg_size * 8
            # if bits not in (8, 16, 32, 64, 128):
            #     bits = 1 << (bits - 1).bit_length()
            # param_types.append(self.ir.IntType(bits))
            # See _arg_ir_type for why wide args use [N x i8].
            param_types.append(self._arg_ir_type(arg_size))

        func_ty = self.ir.FunctionType(self.ir.VoidType(), param_types, False)
        ir_function = self.ir_functions.get(func.name)
        if ir_function is None:
            ir_function = self.ir.Function(llvm_module, func_ty, func.name)
            self.ir_functions[func.name] = ir_function
        elif ir_function.function_type != func_ty:
            raise InvalidTypeException(
                f"Function type mismatch for {func.name}: {ir_function.function_type} vs {func_ty}"
            )

        if self.arg_map and func.name in self.arg_map:
            self._mark_kernel(llvm_module, ir_function)
        elif not self.arg_map:
            # SASS-only mode (no ELF -> no KPARAM map): the function takes no
            # IR params (the code reads the captured blob through the
            # @const_mem global), but it must still be a KERNEL (.entry) or
            # cuModuleGetFunction can't find it and the GPU-launch drivers
            # (test/common/kernel_wrapper_cuda.h, cuDNN CD_* benches) fail.
            self._mark_kernel(llvm_module, ir_function)

        func.block_map = {}
        func.build_bb_to_ir_map(ir_function, func.block_map)

        entry_block = func.block_map[func.blocks[0]]
        builder = self.ir.IRBuilder(entry_block)
        builder.position_at_start(entry_block)

        local_array_ty = self.ir.ArrayType(self.ir.IntType(8), 32768)
        self._local_mem = builder.alloca(local_array_ty, name="local_mem")
        # 16-byte aligned so ptxas-emitted wide local accesses (LDL.128/STL.128
        # → ld/st.local.v2.b64 / v4.b32) don't fault with MISALIGNED_ADDRESS.
        self._local_mem.align = 16

        # Per-thread alloca for the marshalled kernel-arg region (const bank 0).
        # Sized to cover every arg of this kernel (16B-padded for ptxas wide
        # loads/stores). Replaces the contended global @const_mem for params.
        param_size = max((off + sz for off, sz in ordered_args), default=0)
        param_size = max((param_size + 15) & ~15, 256)
        self._param_local = builder.alloca(
            self.ir.ArrayType(self.ir.IntType(8), param_size), name="param_local")
        self._param_local.align = 16

        for idx, (arg_offset, arg_size) in enumerate(ordered_args):
            ir_arg = ir_function.args[idx]
            ir_arg.name = f"arg_{arg_offset}"
            self._store_arg_to_const_mem(builder, ir_arg, arg_offset, arg_size)

        ir_regs = {}

        for bb in func.blocks:
            ir_block = func.block_map[bb]
            builder = self.ir.IRBuilder(ir_block)
            self._lift_basic_block(bb, builder, ir_regs, func.block_map)

        for bb in func.blocks:
            builder = self.ir.IRBuilder(func.block_map[bb])
            self._populate_phi_nodes(bb, builder, ir_regs, func.block_map)

        self._ensure_block_terminators(func)

    def _atom_add_f16x2(self, IRBuilder, ptr, src_val):
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        addr_int = IRBuilder.ptrtoint(ptr, i64, "f16x2_addr")
        asm_ty = self.ir.FunctionType(i32, [i64, i32])
        # side_effect=True: the atom result ($0) is usually unused, and without
        # the side-effect flag LLVM treats the asm call as pure and DCEs the
        # entire atomic (f16atomic: FP16 add silently dropped -> result stayed 0).
        inline_asm = self.ir.InlineAsm(
            asm_ty,
            "atom.global.add.noftz.f16x2 $0, [$1], $2;",
            "=r,l,r,~{memory}",
            side_effect=True,
        )
        return IRBuilder.call(inline_asm, [addr_int, src_val], "atom_f16x2_add")

    # SASS-op (ADD/MIN/MAX/AND/OR/XOR/EXCH/CAS) -> nvvm sys-atomic intrinsic stem.
    _SYS_ATOM_INTR = {
        "ADD": "add", "MIN": "min", "MAX": "max", "AND": "and",
        "OR": "or", "XOR": "xor", "EXCH": "exch", "CAS": "cas",
    }

    def _atom_sys_intrinsic(self, IRBuilder, ptr, op_type, src_val,
                            cmp_val=None, name="atom_sys"):
        """Emit a SYSTEM-scope integer atomic via `llvm.nvvm.atomic.*.gen.i.sys`
        (-> `atom.sys.*` PTX). SASS `.SYS`-scoped ATOMG/REDG carry system scope
        for cross-device (host<->GPU on managed memory) coherence. A plain
        `atomicrmw` lowers to DEVICE scope (`atom.global.*`, no `.sys`) on NVPTX,
        so concurrent host updates race and increments are LOST (atomicSystemWide:
        testData < expected). The intrinsic takes a GENERIC (addrspace-0) pointer;
        cast the addrspace(1) ptr to generic — the HW resolves it at runtime.

        Only the integer ops are covered (the `.gen.i.sys` intrinsic family);
        float/INC/DEC `.SYS` atomics fall back to the device-scope atomicrmw path
        (none appear in the current SASS corpus)."""
        stem = self._SYS_ATOM_INTR.get(op_type)
        if stem is None:
            return None
        elem_ty = ptr.type.pointee
        if not isinstance(elem_ty, self.ir.IntType):
            return None
        width = elem_ty.width
        if width not in (32, 64):
            return None
        gen_ptr = IRBuilder.addrspacecast(
            ptr, self.ir.PointerType(elem_ty, 0), f"{name}_gen")
        intr_name = f"llvm.nvvm.atomic.{stem}.gen.i.sys.i{width}.p0"
        cached = self.intrinsics.get(intr_name)
        if cached is None:
            arg_tys = [self.ir.PointerType(elem_ty, 0), elem_ty]
            if op_type == "CAS":
                arg_tys.append(elem_ty)
            func_ty = self.ir.FunctionType(elem_ty, arg_tys, False)
            cached = self.ir.Function(self.llvm_module, func_ty, intr_name)
            self.intrinsics[intr_name] = cached
        args = [gen_ptr, src_val]
        if op_type == "CAS":
            args = [gen_ptr, cmp_val, src_val]
        return IRBuilder.call(cached, args, name)

    def _fixup_generic_shared_addr(self, builder, addr_i64, name):
        """Bias a generic (addrspace-0) integer address that targets the
        __shared__ window so it lands at the correct @shared_mem byte.

        The SASS materializes a generic pointer to a __shared__ object as
        `(S2R SR_SWINHI) << 32 | LEA((CgaCtaId<<24) + raw_off)` (quicksort's
        lqsort `__shared__ T *s = mys`). SR_SWINHI is lifted to
        `cvta.shared(@shared_mem) >> 32`, so the HIGH word already carries the
        shared-window base. But `raw_off` is the @shared_mem-RELATIVE byte
        offset, whereas on sm_90 ptxas places @shared_mem at a NONZERO window
        offset W (the 0x400 driver-reserved region at the start of CTA shared:
        `cvta.shared(@shared_mem) & 0xffffffff == W`). The OR construction drops
        W, so the generic pointer ends up W bytes BELOW the data: a generic load
        reads @shared_mem[raw_off - W] instead of @shared_mem[raw_off]
        (quicksort: `s[i]` read mys[i-256] -> wrong values / multi-MB OOB).

        Add W back, but ONLY when the address actually lies in the shared window
        (its high 32 bits equal cvta.shared(@shared_mem)'s high word). A genuine
        global generic address has a different high word and is left untouched.
        This keeps the generic load addrspace-agnostic (the HW still resolves it
        at runtime) while making the shared round-trip self-consistent with the
        direct LDS/STS accesses, which use @shared_mem byte offsets directly.

        CLUSTER (sm_90 DSMEM) caveats, from FA3 fwd PackGQA (its Q gather does
        generic LD.E.128 from a PEER CTA's shared window; dense FA never
        generic-loads DSMEM so this went unseen):
        1. W must be RANK-STRIPPED: on a cluster launch cvta.shared(@shared_mem)
           low word = (own_ctarank<<24) | window_off. Adding the own-rank byte
           to a peer address `(peer_rank<<24)|off` corrupts the rank field
           (rank 1+1(+carry) -> "CTA 3" faults in a 2-CTA cluster).
        2. When @shared_mem models the DYNAMIC region, the SASS offsets are
           whole-window-based (`PRMT(rank,0x400,0x654)` = dyn byte 0 at window
           0x400), not @shared_mem-relative — subtract _shared_dyn_base so the
           bias is only the difference between OUR window placement and the
           original's. Static-shared modules (_shared_dyn_base == 0, quicksort)
           keep the original raw-offset semantics unchanged.
        """
        i64 = self.ir.IntType(64)
        sm_i8 = builder.bitcast(
            self.SharedMem, self.ir.PointerType(self.ir.IntType(8), 3),
            name + "_smfix_i8")
        gen = builder.addrspacecast(
            sm_i8, self.ir.PointerType(self.ir.IntType(8), 0), name + "_smfix_gen")
        base = builder.ptrtoint(gen, i64, name + "_smfix_base")
        w = builder.and_(base, self.ir.Constant(i64, 0x00FFFFFF), name + "_smfix_w")
        dyn = getattr(self, "_shared_dyn_base", 0)
        if dyn:
            w = builder.sub(w, self.ir.Constant(i64, dyn), name + "_smfix_wdyn")
        base_hi = builder.lshr(base, self.ir.Constant(i64, 32), name + "_smfix_basehi")
        addr_hi = builder.lshr(addr_i64, self.ir.Constant(i64, 32), name + "_smfix_addrhi")
        in_window = builder.icmp_unsigned(
            "==", addr_hi, base_hi, name + "_smfix_inwin")
        biased = builder.add(addr_i64, w, name + "_smfix_biased")
        return builder.select(in_window, biased, addr_i64, name + "_smfix")

    def _addrspace_pointer(self, builder, value, pointee_ty, addrspace, name):
        tgt_ptr_ty = self.ir.PointerType(pointee_ty, addrspace)
        ty = value.type
        if ty.is_pointer:
            src_as = ty.addrspace
            if src_as == addrspace:
                if ty.pointee == pointee_ty:
                    return value
                return builder.bitcast(value, tgt_ptr_ty, name)
            if ty.pointee == pointee_ty:
                return builder.addrspacecast(value, tgt_ptr_ty, name)
            same_as_ptr = builder.bitcast(
                value,
                self.ir.PointerType(pointee_ty, src_as),
                f"{name}_bitcast_srcas",
            )
            return builder.addrspacecast(same_as_ptr, tgt_ptr_ty, name)

        if ty.intrinsic_name[0] != 'i':
            raise InvalidTypeException(
                f"Expected integer address for {name}, got {ty}"
            )

        if ty.width < 64:
            value = builder.zext(
                value, self.ir.IntType(64), f"{name}_zext64"
            )
        elif ty.width > 64:
            value = builder.trunc(
                value, self.ir.IntType(64), f"{name}_trunc64"
            )

        # A generic (addrspace-0) integer address may be a materialized pointer
        # into the __shared__ window (SR_SWINHI construction); rebias it onto
        # @shared_mem's actual window offset. No-op for true global addresses.
        if addrspace == 0:
            value = self._fixup_generic_shared_addr(builder, value, name)

        return builder.inttoptr(value, tgt_ptr_ty, name)

    def _get_val(self, builder, ir_regs, op, name=""):
        if op.is_rz:
            return self.ir.Constant(op.get_ir_type(self), 0)

        if op.is_pt:
            # int(...) — bare Python bool here makes llvmlite emit
            # `i32 true` (declared type i32 but stringified as bool),
            # which llc rejects with a constant-expression-type-mismatch.
            return self.ir.Constant(op.get_ir_type(self), int(not op.is_not_reg))

        if op.is_immediate:
            ty = op.get_ir_type(self)
            val = op.immediate_value
            if isinstance(ty, self.ir.IntType) and isinstance(val, float):
                import struct
                if ty.width == 32:
                    val = struct.unpack('<I', struct.pack('<f', val))[0]
                elif ty.width == 64:
                    val = struct.unpack('<Q', struct.pack('<d', val))[0]
                elif ty.width == 16:
                    import numpy as np
                    val = int(np.float16(val).view(np.uint16))
            elif isinstance(ty, (self.ir.FloatType, self.ir.DoubleType, self.ir.HalfType)) and isinstance(val, int):
                # An int-bits immediate (e.g. `MOV Rf, 0x3f49234e`) into a float
                # operand carries the float's raw bits — reinterpret, don't convert.
                import struct
                if isinstance(ty, self.ir.FloatType):
                    val = struct.unpack('<f', struct.pack('<I', val & 0xffffffff))[0]
                elif isinstance(ty, self.ir.DoubleType):
                    val = struct.unpack('<d', struct.pack('<Q', val & 0xffffffffffffffff))[0]
                else:  # HalfType
                    import numpy as np
                    val = float(np.uint16(val & 0xffff).view(np.float16))
            return self.ir.Constant(ty, val)

        if op.is_mem_addr and not op.is_reg:  # register-less address, e.g. LDS.U R2, [0xfc]
            # The literal address lives in offset_value; is_mem_addr operands are
            # never flagged is_immediate (see operand.py parsing).
            return self.ir.Constant(op.get_ir_type(self), op.offset_value)

        if op.is_const_mem:
            val = self._get_const_mem_val(builder, ir_regs, op, name)

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
            val = ir_regs[op.get_ir_name(self)]
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
        # SASS operand-modifier order is abs (|.|) FIRST, then negate (-): the
        # token `-|R|` means -(|R|). Applying negate before abs makes
        # fabs(fneg(val)) = |val|, silently DROPPING the negate (atan2's
        # `|x| - |y|` numerator became `|x| + |y|` -> result 1.0 not 0, NaN/
        # garbage). Apply abs first, then negate.
        if op.is_abs_reg:
            if type_desc in FLOAT64:
                val = self.run_intrinsic("fabs_f64", builder, [val], f"{name}_fabs")
            elif type_desc in FLOAT32:
                val = self.run_intrinsic("fabs", builder, [val], f"{name}_fabs")
            else:
                val = self.run_intrinsic("abs", builder, [val], f"{name}_abs")
        if op.is_not_reg:
            val = builder.not_(val, f"{name}_not")
        if op.is_negative_reg:
            if type_desc.startswith("F"):
                val = builder.fneg(val, f"{name}_fneg")
            else:
                val = builder.neg(val, f"{name}_neg")
                
        if op.suffix and ("H0" in op.suffix or "H1" in op.suffix):
            orders = op.suffix.split("_")
            i32 = self.ir.IntType(32)
            if not isinstance(val.type, self.ir.IntType):
                val = self._insert_type_cast_if_needed(builder, val, i32, f"{name}_to_i32")
            h0 = builder.and_(val, self.ir.Constant(val.type, 0xFFFF), f"{name}_h0" if name else "h0")
            h1 = builder.and_(
                builder.lshr(
                    val,
                    self.ir.Constant(val.type, 16),
                    f"{name}_h1_shift" if name else "h1_shift",
                ),
                self.ir.Constant(val.type, 0xFFFF),
                f"{name}_h1" if name else "h1",
            )
            if len(orders) == 1:
                val = h0 if orders[0] == "H0" else h1
            else:
                low = h0 if orders[0] == "H0" else h1
                high = h0 if orders[1] == "H0" else h1
                val = builder.or_(
                    low,
                    builder.shl(
                        high,
                        self.ir.Constant(val.type, 16),
                        f"{name}_h_high_shift" if name else "h_high_shift",
                    ),
                    f"{name}_h_pack" if name else "h_pack",
                )

            

        if op.is_mem_addr and op.offset:
            val = builder.add(
                val, self.ir.Constant(val.type, op.offset), f"{name}_memaddr_offset_add"
            )

        return val

    def _as_pointer(self, builder, addr_val, pointee_ty, name, addrspace=0):
        return self._addrspace_pointer(builder, addr_val, pointee_ty, addrspace, name)

    def _mark_kernel(self, module, fn):
        try:
            md_node = module.add_metadata(
                [
                    fn,
                    ir.MetaDataString(module, "kernel"),
                    ir.Constant(ir.IntType(32), 1),
                ]
            )
            module.add_named_metadata("nvvm.annotations").add(md_node)

            # Carry the original kernel's launch bounds (EIATTR_MAX_THREADS) as
            # nvvm maxntid*, so ptxas caps register usage to fit the block size.
            # Without this, ptxas over-allocates registers (the lift has no
            # bounds) and a large-block launch — e.g. the 1024-thread nvtiff LZW
            # decode or NPP resize — fails with "too many resources requested".
            mt = None
            if self.elf is not None and hasattr(self.elf, "max_threads_for"):
                mt = self.elf.max_threads_for(fn.name)
            if mt:
                i32 = ir.IntType(32)
                for axis, dim in zip(("maxntidx", "maxntidy", "maxntidz"), mt):
                    if dim and dim > 0:
                        node = module.add_metadata(
                            [fn, ir.MetaDataString(module, axis),
                             ir.Constant(i32, int(dim))]
                        )
                        module.add_named_metadata("nvvm.annotations").add(node)
        except Exception as e:
            print("Warning: failed to add nvvm.annotations metadata:", e)


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
    
    @lift_for("S2R")
    def lift_s2r(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        valop = Inst.get_uses()[0]

        def _call_sreg(sreg_name, tag):
            return self.run_intrinsic(f"sreg_{sreg_name}", IRBuilder, [], tag)

        if valop.is_thread_idx_x:
            val = _call_sreg("tid.x", "tid_x")
        elif valop.is_thread_idx_y:
            val = _call_sreg("tid.y", "tid_y")
        elif valop.is_thread_idx_z:
            val = _call_sreg("tid.z", "tid_z")
        elif valop.is_block_dim_x:
            val = _call_sreg("ntid.x", "ntid_x")
        elif valop.is_block_dim_y:
            val = _call_sreg("ntid.y", "ntid_y")
        elif valop.is_block_dim_z:
            val = _call_sreg("ntid.z", "ntid_z")
        elif valop.is_block_idx_x:
            val = _call_sreg("ctaid.x", "ctaid_x")
        elif valop.is_block_idx_y:
            val = _call_sreg("ctaid.y", "ctaid_y")
        elif valop.is_block_idx_z:
            val = _call_sreg("ctaid.z", "ctaid_z")
        elif valop.is_grid_dim_x:
            val = _call_sreg("nctaid.x", "nctaid_x")
        elif valop.is_grid_dim_y:
            val = _call_sreg("nctaid.y", "nctaid_y")
        elif valop.is_grid_dim_z:
            val = _call_sreg("nctaid.z", "nctaid_z")
        elif valop.is_lane_id:
            val = _call_sreg("laneid", "laneid")
        elif valop.is_warp_id:
            val = _call_sreg("warpid", "warpid")
        elif valop.is_warp_size:
            val = _call_sreg("warpsize", "warpsize")
        elif valop.is_active_mask:
            # llvm.nvvm.activemask -> PTX activemask.b32. There is no
            # %sreg for activemask (the sreg-named intrinsic would lower
            # to an unresolvable extern call).
            val = self.run_intrinsic("activemask", IRBuilder, [], "activemask")
        elif valop.is_lane_mask_eq:
            val = _call_sreg("lanemask.eq", "lanemask_eq")
        elif valop.is_lane_mask_le:
            val = _call_sreg("lanemask.le", "lanemask_le")
        elif valop.is_lane_mask_lt:
            val = _call_sreg("lanemask.lt", "lanemask_lt")
        elif valop.is_lane_mask_ge:
            val = _call_sreg("lanemask.ge", "lanemask_ge")
        elif valop.is_lane_mask_gt:
            val = _call_sreg("lanemask.gt", "lanemask_gt")
        elif valop.is_clock_lo:
            val = _call_sreg("clock", "clock_lo")
        elif valop.is_clock_hi:
            # NVPTX has no direct clockhi sreg; clock64 returns full i64.
            # Read clock64 and shift-right-32+trunc to get the high half.
            clk64 = self.run_intrinsic("sreg_clock64", IRBuilder, [], "clock64_for_hi")
            shifted = IRBuilder.lshr(clk64, self.ir.Constant(self.ir.IntType(64), 32), "clock_hi_shr")
            val = IRBuilder.trunc(shifted, self.ir.IntType(32), "clock_hi")
        elif valop.is_global_timer_lo:
            val = _call_sreg("globaltimer.lo", "gtimer_lo")
        elif valop.is_global_timer_hi:
            val = _call_sreg("globaltimer.hi", "gtimer_hi")
        elif valop.is_cluster_ctarank:
            # SM90 SR_CgaCtaId -> %cluster_ctarank (CTA rank within cluster).
            # Must be the real sreg, not constant 0: folding it lets LLVM
            # restructure the divergent regions around vote.sync and the
            # resulting (valid) PTX SIGSEGVs ptxas 12.2 at -O1+ in the
            # reduced cluster reproducer. Non-cluster launches read 0 at
            # runtime, so semantics match the SASS either way.
            val = _call_sreg("cluster.ctarank", "cluster_ctarank")
        elif valop.name.startswith("SR_SWINLO") or valop.name.startswith("SR_SWINHI"):
            # Generic base address of the shared-memory window. Kernels that
            # materialize a generic pointer to __shared__ data (quicksort's
            # lqsort `__shared__ T *s`) build (SWINHI<<32 | SWINLO + offset)
            # then dereference with generic LD/ST. Provide cvta.shared
            # (@shared_mem) lo/hi so those generic accesses resolve back into
            # our shared_mem array (offsets preserved from the cubin layout).
            val = self._swin_value(
                IRBuilder, valop.name.startswith("SR_SWINHI"), "swin")
        else:
            print(f"S2R: Unknown special register {valop.name}")
            val = self.ir.Constant(self.ir.IntType(32), 0)

        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("S2UR")
    def lift_s2ur(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        valop = Inst.get_uses()[0]

        def _call_sreg(sreg_name, tag):
            return self.run_intrinsic(f"sreg_{sreg_name}", IRBuilder, [], tag)

        if valop.is_thread_idx_x:
            val = _call_sreg("tid.x", "s2ur_tid_x")
        elif valop.is_thread_idx_y:
            val = _call_sreg("tid.y", "s2ur_tid_y")
        elif valop.is_thread_idx_z:
            val = _call_sreg("tid.z", "s2ur_tid_z")
        elif valop.is_block_dim_x:
            val = _call_sreg("ntid.x", "s2ur_ntid_x")
        elif valop.is_block_dim_y:
            val = _call_sreg("ntid.y", "s2ur_ntid_y")
        elif valop.is_block_dim_z:
            val = _call_sreg("ntid.z", "s2ur_ntid_z")
        elif valop.is_block_idx_x:
            val = _call_sreg("ctaid.x", "s2ur_ctaid_x")
        elif valop.is_block_idx_y:
            val = _call_sreg("ctaid.y", "s2ur_ctaid_y")
        elif valop.is_block_idx_z:
            val = _call_sreg("ctaid.z", "s2ur_ctaid_z")
        elif valop.is_grid_dim_x:
            val = _call_sreg("nctaid.x", "s2ur_nctaid_x")
        elif valop.is_grid_dim_y:
            val = _call_sreg("nctaid.y", "s2ur_nctaid_y")
        elif valop.is_grid_dim_z:
            val = _call_sreg("nctaid.z", "s2ur_nctaid_z")
        elif valop.is_lane_id:
            val = _call_sreg("laneid", "s2ur_laneid")
        elif valop.is_warp_id:
            val = _call_sreg("warpid", "s2ur_warpid")
        elif valop.is_warp_size:
            val = _call_sreg("warpsize", "s2ur_warpsize")
        elif valop.is_active_mask:
            # See lift_s2r: activemask is not a %sreg.
            val = self.run_intrinsic("activemask", IRBuilder, [], "s2ur_activemask")
        elif valop.is_lane_mask_eq:
            val = _call_sreg("lanemask.eq", "s2ur_lanemask_eq")
        elif valop.is_lane_mask_le:
            val = _call_sreg("lanemask.le", "s2ur_lanemask_le")
        elif valop.is_lane_mask_lt:
            val = _call_sreg("lanemask.lt", "s2ur_lanemask_lt")
        elif valop.is_lane_mask_ge:
            val = _call_sreg("lanemask.ge", "s2ur_lanemask_ge")
        elif valop.is_lane_mask_gt:
            val = _call_sreg("lanemask.gt", "s2ur_lanemask_gt")
        elif valop.is_clock_lo:
            val = _call_sreg("clock", "s2ur_clock_lo")
        elif valop.is_clock_hi:
            clk64 = self.run_intrinsic("sreg_clock64", IRBuilder, [], "s2ur_clock64_for_hi")
            shifted = IRBuilder.lshr(clk64, self.ir.Constant(self.ir.IntType(64), 32), "s2ur_clock_hi_shr")
            val = IRBuilder.trunc(shifted, self.ir.IntType(32), "s2ur_clock_hi")
        elif valop.is_global_timer_lo:
            val = _call_sreg("globaltimer.lo", "s2ur_gtimer_lo")
        elif valop.is_global_timer_hi:
            val = _call_sreg("globaltimer.hi", "s2ur_gtimer_hi")
        elif valop.is_cluster_ctarank:
            # SM90 SR_CgaCtaId -> %cluster_ctarank; see lift_s2r. (S2UR reads
            # it into a uniform register — the rank is uniform across the CTA.)
            val = _call_sreg("cluster.ctarank", "s2ur_cluster_ctarank")
        elif valop.name.startswith("SR_SWINLO") or valop.name.startswith("SR_SWINHI"):
            # Shared-window generic base — see lift_s2r / _swin_value. FA3 /
            # CUTLASS Hopper kernels read it via S2UR (uniform across the CTA)
            # to build generic pointers into shared and to cvta generic
            # addresses back to window offsets; falling through to constant 0
            # made every such address garbage (out-of-range shared).
            val = self._swin_value(
                IRBuilder, valop.name.startswith("SR_SWINHI"), "s2ur_swin")
        else:
            print(f"S2UR: Unknown special register {valop.name}")
            val = self.ir.Constant(self.ir.IntType(32), 0)

        self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("LDG", "LDG64", "LD")
    def lift_ldg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldg_addr")
        dest_ty = dest.get_ir_type(self)

        # LDG/LDG64 are global loads (addrspace 1); LD (no G) is a generic load
        # (addrspace 0) the hardware resolves at runtime. This mirrors lift_stg's
        # STG-vs-ST split. quicksort's lqsort dereferences __shared__ T* runtime
        # pointers (cvta.shared generic addresses of mys/mysn) via generic LD.E;
        # forcing addrspace 1 reinterprets those shared-window addresses as global
        # pointers -> ~3.4GB OOB read. ld.global token in the STRONG inline-asm is
        # likewise gated on the addrspace below.
        addrspace = 1 if Inst.opcodes[0] in ("LDG", "LDG64") else 0
        space_tok = ".global" if addrspace == 1 else ""

        # Wide vectorized load into a register group (LDG.E.64/.128 -> Rn:Rn+1[:..]).
        # Loading a wide *integer* (i64/i128) makes NVPTX lower to ld.global.v2.b64
        # + a `mov.b64 {_, hi}` high-word extract. The in-process driver JIT
        # (cuModuleLoad on PTX text, used by the ptxswap correctness harness)
        # MISCOMPILES that extract on sm_90: the high lane of the v2.b64 load reads
        # the *low* lane's value, silently feeding a wrong float4 component into
        # downstream math (clenergy: dy used .x instead of .y, so dist^2 was off by
        # atom.x-atom.z). Standalone ptxas compiles the same PTX correctly at every
        # -O level, so this is purely a driver-JIT codegen bug. Emit the load as a
        # vector of i32 lanes (<N x i32>) so NVPTX uses ld.global.v{2,4}.b32 + per-
        # lane extractelement (which the JIT lowers correctly) and write each
        # 32-bit lane straight into its sub-register, skipping the wide-int
        # shift/trunc round-trip entirely.
        if dest.is_reg_group and not (
            isinstance(dest_ty, self.ir.IntType) and dest_ty.width <= 32
        ):
            n = len(dest.reg_group)
            i32 = self.ir.IntType(32)
            # STRONG-ordered wide load (LD.E.128.STRONG.GPU): the decoupled-
            # lookback ScanTileState read (CUB reduce-by-key/scan) — a 128-bit
            # {status,value} fetched in the look-back spin loop. A plain vector
            # load is loop-invariant -> LLVM/ptxas hoists it out of the spin loop
            # -> the wait reads stale/zero tile state forever -> hang or all-zero
            # aggregates (segment-reduce). Emit a volatile per-lane vector load so
            # it stays in the loop, mirroring the scalar STRONG path below.
            if "STRONG" in [t.upper() for t in Inst.opcodes[1:]] and n in (2, 4):
                i64 = self.ir.IntType(64)
                lanes = ",".join(f"${k}" for k in range(n))
                vec_struct = self.ir.LiteralStructType([i32] * n)
                asm_ty = self.ir.FunctionType(vec_struct, [i64])
                addr_int = IRBuilder.ptrtoint(
                    _as_pointer(
                        addr, i32,
                        f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr",
                        addrspace=addrspace),
                    i64, "ldstrongv_addr")
                ld_asm = self.ir.InlineAsm(
                    asm_ty,
                    f"ld.volatile{space_tok}.v{n}.b32 {{{lanes}}}, [${n}];",
                    ("=r," * n) + "l,~{memory}", side_effect=True,
                )
                raw = IRBuilder.call(ld_asm, [addr_int], "ldv_volatile")
                for i, sub_op in enumerate(dest.reg_group):
                    IRRegs[sub_op.get_ir_name(self)] = IRBuilder.extract_value(
                        raw, i, f"ldstrongv_lane{i}")
                return
            vec_ty = self.ir.VectorType(i32, n)
            vaddr_ptr = _as_pointer(
                addr,
                vec_ty,
                f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr",
                addrspace=addrspace,
            )
            vec = IRBuilder.load(vaddr_ptr, "ldg_vec", typ=vec_ty)
            for i, sub_op in enumerate(dest.reg_group):
                IRRegs[sub_op.get_ir_name(self)] = IRBuilder.extract_element(
                    vec, self.ir.Constant(i32, i), f"ldg_lane{i}"
                )
            return

        # Narrow-width modifiers: SASS LDG.E.U8/S8/U16/S16 loads 1 or 2 bytes
        # from memory and zero/sign-extends into the (32-bit) destination
        # register. Emitting a full-width load would over-read and may fault
        # on a 1- or 2-byte-aligned pointer (CUDA_ERROR_MISALIGNED_ADDRESS).
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        is_signed = False
        if "U8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "S8" in mods:
            narrow_ty = self.ir.IntType(8)
            is_signed = True
        elif "U16" in mods:
            narrow_ty = self.ir.IntType(16)
        elif "S16" in mods:
            narrow_ty = self.ir.IntType(16)
            is_signed = True

        use_narrow = narrow_ty is not None and isinstance(dest_ty, self.ir.IntType)
        pointee_ty = narrow_ty if use_narrow else dest_ty

        addr_ptr = _as_pointer(
            addr,
            pointee_ty,
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "ldg_addr_ptr",
            addrspace=addrspace,
        )
        # STRONG-ordered loads (LD.E.STRONG/.SYS) are synchronizing reads, e.g.
        # the spin-wait flag in decoupled-lookback prefix scan / spin-locks. A
        # plain `load` is loop-invariant to LLVM/ptxas and gets HOISTED out of
        # the spin loop -> the wait reads a stale cached value forever -> hang /
        # ~60x timeout (scan3, pad). llvmlite has no volatile-load, so emit a
        # PTX volatile load via inline asm (not cached, not hoisted). Only the
        # plain 32/64-bit widths; narrow STRONG loads fall through to plain.
        is_strong = "STRONG" in mods
        if is_strong and not use_narrow and isinstance(pointee_ty, (self.ir.IntType, self.ir.FloatType, self.ir.DoubleType)):
            ld_w = pointee_ty.width if isinstance(pointee_ty, self.ir.IntType) else (64 if isinstance(pointee_ty, self.ir.DoubleType) else 32)
            if ld_w in (32, 64):
                i64 = self.ir.IntType(64)
                ld_int_ty = self.ir.IntType(ld_w)
                addr_int = IRBuilder.ptrtoint(addr_ptr, i64, "ldstrong_addr")
                rc = "l" if ld_w == 64 else "r"
                asm_ty = self.ir.FunctionType(ld_int_ty, [i64])
                ld_asm = self.ir.InlineAsm(
                    asm_ty, f"ld.volatile{space_tok}.b{ld_w} $0, [$1];",
                    f"={rc},l,~{{memory}}", side_effect=True,
                )
                raw = IRBuilder.call(ld_asm, [addr_int], "ld_volatile")
                val = raw if ld_int_ty == pointee_ty else IRBuilder.bitcast(raw, pointee_ty, "ldstrong_cast")
            else:
                val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
        else:
            val = IRBuilder.load(addr_ptr, "ldg", typ=pointee_ty)
        if use_narrow and dest_ty != narrow_ty:
            if is_signed:
                val = IRBuilder.sext(val, dest_ty, "ldg_sext")
            else:
                val = IRBuilder.zext(val, dest_ty, "ldg_zext")
        self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("STG", "ST")
    def lift_stg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "stg_addr")
        v = _get_val(val, "stg_val")

        # Narrow-width modifiers: STG.E.U8/S8/U16/S16 truncates the source
        # register to 1 or 2 bytes before storing. See lift_ldg for the
        # alignment rationale.
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        if "U8" in mods or "S8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "U16" in mods or "S16" in mods:
            narrow_ty = self.ir.IntType(16)
        if narrow_ty is not None and isinstance(v.type, self.ir.IntType) and v.type.width > narrow_ty.width:
            v = IRBuilder.trunc(v, narrow_ty, "stg_trunc")

        pointee_ty = v.type
        # STG is a global store (addrspace 1); ST (no G) is a generic store
        # (addrspace 0) the hardware resolves at runtime.
        addrspace = 1 if Inst.opcodes[0] == "STG" else 0
        addr_ptr = _as_pointer(
            addr,
            pointee_ty,
            f"{ptr.get_ir_name(self)}_addr_ptr" if ptr.is_reg else "stg_addr_ptr",
            addrspace=addrspace,
        )
        # STRONG-ordered store (ST.E.*.STRONG.GPU): the release-side publish of a
        # decoupled-lookback ScanTileState ({status,value}) in CUB reduce-by-key/
        # scan. The matching look-back load is volatile (see lift_ldg); the publish
        # must be volatile too or ptxas may sink/cache it so partner tiles never
        # observe the aggregate (segment-reduce all-zero sums). Emit a volatile PTX
        # store for int widths 32/64/128.
        if ("STRONG" in mods and isinstance(pointee_ty, self.ir.IntType)
                and pointee_ty.width in (32, 64, 128)):
            i64 = self.ir.IntType(64)
            addr_int = IRBuilder.ptrtoint(addr_ptr, i64, "ststrong_addr")
            w = pointee_ty.width
            if w == 128:
                lo = IRBuilder.trunc(v, i64, "ststrong_lo")
                hi = IRBuilder.trunc(
                    IRBuilder.lshr(v, self.ir.Constant(pointee_ty, 64), "ststrong_hish"),
                    i64, "ststrong_hi")
                asm_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i64, i64])
                st_asm = self.ir.InlineAsm(
                    asm_ty, "st.volatile.global.v2.b64 [$0], {$1,$2};",
                    "l,l,l,~{memory}", side_effect=True,
                )
                IRBuilder.call(st_asm, [addr_int, lo, hi], "")
            else:
                rc = "l" if w == 64 else "r"
                asm_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, pointee_ty])
                st_asm = self.ir.InlineAsm(
                    asm_ty, f"st.volatile.global.b{w} [$0], $1;",
                    f"l,{rc},~{{memory}}", side_effect=True,
                )
                IRBuilder.call(st_asm, [addr_int, v], "")
        else:
            IRBuilder.store(v, addr_ptr)


    def _swin_value(self, IRBuilder, want_hi, tag):
        """Value of SR_SWINLO / SR_SWINHI: generic-space base of the shared
        window, as the low/high 32 bits.

        Modeled as cvta.shared of @shared_mem[0], minus the dynamic-window
        rebase. The SASS addresses shared bytes at WINDOW offsets (dynamic
        data starts at `_shared_dyn_base`, e.g. the sm_90 1 KiB reserved
        region), while @shared_mem[0] is dynamic byte 0. Both idioms the SASS
        builds from SWIN must land on the same cell:

          generic deref:  SWIN + window_off        (direct LD/ST)
          round-trip:     _shared_gep(gen - SWINLO) = gen - SWIN
                          - _shared_dyn_base       (cvta.to.shared pattern)

        With SWIN = generic(@shared_mem[0]) - _shared_dyn_base, both resolve
        to @shared_mem[window_off - _shared_dyn_base]. In the static-shared
        case _shared_dyn_base is 0 and this is plain cvta.shared (quicksort's
        validated behavior, unchanged).

        CLUSTER caveat (FA3 fwd PackGQA): the real SR_SWINLO is the RANK-FREE
        window base — SASS composes DSMEM addresses as
        SWINLO + (target_rank<<24 | off). cvta.shared(@shared_mem)'s low word
        carries OUR OWN ctarank in bits 24-31, so the lo value must strip
        them or peer addresses get (own+target)<<24 ("CTA 2/3" OOB faults in
        a 2-CTA cluster). Rank bits are zero on non-cluster launches, so the
        mask is a no-op for the validated quicksort behavior.
        """
        i8 = self.ir.IntType(8)
        i64 = self.ir.IntType(64)
        sm_i8 = IRBuilder.bitcast(
            self.SharedMem, self.ir.PointerType(i8, 3), tag + "_sm_i8")
        gen = IRBuilder.addrspacecast(
            sm_i8, self.ir.PointerType(i8, 0), tag + "_gen")
        base = IRBuilder.ptrtoint(gen, i64, tag + "_base")
        dyn = getattr(self, "_shared_dyn_base", 0)
        if dyn:
            base = IRBuilder.sub(
                base, self.ir.Constant(i64, dyn), tag + "_rebase")
        if want_hi:
            base = IRBuilder.lshr(
                base, self.ir.Constant(i64, 32), tag + "_hi_shr")
        else:
            base = IRBuilder.and_(
                base, self.ir.Constant(i64, 0x00FFFFFF), tag + "_lo_norank")
        return IRBuilder.trunc(base, self.ir.IntType(32), tag)

    def _shared_gep(self, IRBuilder, addr, name):
        """GEP into @shared_mem with the dynamic-shared base offset removed.

        When @shared_mem models only the dynamic region (sized at launch), the
        SASS shared index carries the static .nv.shared size as a base offset
        (the `(CgaCtaId<<24)|static_shared` cluster-window pointer). Subtract it
        so the index is relative to the dynamic region's start (byte 0). For the
        static-shared case _shared_dyn_base is 0 and this is a no-op.
        """
        # Strip the cluster-window CTA-rank select bits. sm_90 SASS shared
        # addresses ride the DSMEM window encoding `(CgaCtaId<<24) | offset`
        # (bit 24+ = which CTA's window; ptxas builds the base via
        # UPRMT rank,0x654,0x400 or ULEA). Our flat @shared_mem models the
        # LOCAL CTA's window, so the rank bits must be masked off — real
        # shared offsets never exceed 16 MiB (max 228 KiB on Hopper), so this
        # is a no-op for every non-cluster/rank-0 address and for pre-sm_90
        # ISAs. NOTE: an access whose rank bits name a PEER CTA (explicit
        # DSMEM) would be silently localized here — none observed yet; a
        # faithful peer lowering needs mapa.shared::cluster.
        #
        # The incoming SSA value may be float-typed when the physical register
        # was last written by an FP op on another path (FA3 fwd split: mbarrier
        # address reg reuses a softmax temp) — normalize to i32 bits first.
        addr = self._insert_type_cast_if_needed(
            IRBuilder, addr, self.ir.IntType(32), name + "_i32")
        addr = IRBuilder.and_(
            addr, self.ir.Constant(addr.type, 0xFFFFFF), name + "_winoff"
        )
        base = getattr(self, "_shared_dyn_base", 0)
        if base:
            addr = IRBuilder.sub(
                addr, self.ir.Constant(addr.type, base), name + "_dynbase"
            )
        return IRBuilder.gep(
            self.SharedMem,
            [self.ir.Constant(self.ir.IntType(32), 0), addr],
            name,
        )

    @lift_for("LDTM")
    def lift_ldtm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, **kwargs):
        count = Inst.ldtm_register_count
        destinations = Inst.get_defs()
        sources = Inst.get_uses()
        if not count or len(destinations) != count or len(sources) != 1:
            raise UnsupportedInstructionException(f"Unsupported LDTM form: {Inst}")
        word = self.ir.IntType(32)
        source = sources[0]
        if source.is_rz:
            address = self.ir.Constant(word, source.offset or 0)
        else:
            address = self._insert_type_cast_if_needed(IRBuilder, _get_val(source, "ldtm_addr"), word, "ldtm_addr_bits")
        outputs = ", ".join(f"${index}" for index in range(count))
        operation = (f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32 {{{outputs}}}, [${count}];\n\t"
                     "tcgen05.wait::ld.sync.aligned;")
        result_type = word if count == 1 else self.ir.LiteralStructType([word] * count)
        signature = self.ir.FunctionType(result_type, [word])
        constraints = ",".join(["=r"] * count + ["r", "~{memory}"])
        assembly = self.ir.InlineAsm(signature, operation, constraints, side_effect=True)
        result = IRBuilder.call(assembly, [address], "ldtm")
        for index, dest in enumerate(destinations):
            value = result if count == 1 else IRBuilder.extract_value(result, index)
            self._set_val(IRBuilder, IRRegs, dest, value)

    @lift_for("LDSM")
    def lift_ldsm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """LDSM.16.[M|MT]88.N -> PTX ldmatrix.sync.aligned.xN.m8n8[.trans].

        Warp-collective 8x8 16-bit matrix load — the dual of STSM below: each
        lane supplies one 16-byte row address, the N dest regs receive
        fragments GATHERED ACROSS THE WARP in mma layout. Lifting this as a
        scalar LDS (the old path) loads the right 16 bytes into the wrong
        lanes: FA3 bwd hdim96 fills its V_in_regs A-fragments this way and
        the misdistribution skewed dQ/dK by ~8e-2 while V==1 diagnostics
        (lane-invariant) came out exact.
        """
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        count = int(Inst.opcodes[-1]) if Inst.opcodes[-1].isdigit() else 1
        trans = any("MT88" in m for m in Inst.opcodes)

        addr = _get_val(ptr, "ldsm_addr")
        gep = self._shared_gep(IRBuilder, addr, "ldsm_shared")
        addr64 = IRBuilder.ptrtoint(gep, i64, "ldsm_ptr")

        t = ".trans" if trans else ""
        regs = "{" + ",".join(f"${i}" for i in range(count)) + "}"
        asm_str = (f"ldmatrix.sync.aligned.m8n8.x{count}{t}.shared.b16 "
                   f"{regs}, [${count}];")
        cons = ",".join(["=r"] * count) + ",l,~{memory}"
        ret_ty = self.ir.LiteralStructType([i32] * count)
        fn_ty = self.ir.FunctionType(ret_ty, [i64])
        asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
        res = IRBuilder.call(asm, [addr64], "ldsm")

        if count == 1:
            val = IRBuilder.extract_value(res, 0, "ldsm_w0")
        else:
            total_ty = self.ir.IntType(count * 32)
            val = self.ir.Constant(total_ty, 0)
            for k in range(count):
                w = IRBuilder.extract_value(res, k, f"ldsm_w{k}")
                w64 = IRBuilder.zext(w, total_ty, f"ldsm_w{k}_z")
                if k:
                    w64 = IRBuilder.shl(w64, self.ir.Constant(total_ty, 32 * k),
                                        f"ldsm_w{k}_s")
                val = IRBuilder.or_(val, w64, f"ldsm_pack{k}")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("LDS")
    def lift_lds(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "lds_addr")
        dest_ty = dest.get_ir_type(self)

        # Narrow-width modifiers: LDS.U8/S8/U16/S16 — see lift_ldg.
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        is_signed = False
        if "U8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "S8" in mods:
            narrow_ty = self.ir.IntType(8); is_signed = True
        elif "U16" in mods:
            narrow_ty = self.ir.IntType(16)
        elif "S16" in mods:
            narrow_ty = self.ir.IntType(16); is_signed = True
        use_narrow = narrow_ty is not None and isinstance(dest_ty, self.ir.IntType)
        pointee_ty = narrow_ty if use_narrow else dest_ty

        addr = self._shared_gep(IRBuilder, addr, "lds_shared_addr")
        if addr.type.pointee != pointee_ty:
            addr = IRBuilder.bitcast(
                addr,
                self.ir.PointerType(pointee_ty, 3),
                "lds_shared_addr_cast",
            )
        val = IRBuilder.load(addr, "lds", typ=pointee_ty)
        # Volatile so llc -O2 can't forward/elide warp-synchronous shared reads
        # (split's volatile __shared__ scan; see _patch_volatile_memops).
        val.volatile = True
        if use_narrow and dest_ty != narrow_ty:
            if is_signed:
                val = IRBuilder.sext(val, dest_ty, "lds_sext")
            else:
                val = IRBuilder.zext(val, dest_ty, "lds_zext")
        self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("STAS")
    def lift_stas(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """Lower scalar STAS into a DSMEM store plus its PTX completion action."""
        uses = Inst.get_uses()
        if Inst.opcodes != ["STAS"] or len(uses) != 3:
            raise UnsupportedInstructionException("expected scalar STAS [Rn.64+offset], Rt")
        i32 = self.ir.IntType(32)
        mapped = []
        for operand, tag in ((uses[0], "stas_dst"), (uses[2], "stas_mbar")):
            address = self._insert_type_cast_if_needed(IRBuilder, _get_val(operand, tag), i32, tag + "_bits")
            local = IRBuilder.ptrtoint(self._shared_gep(IRBuilder, address, tag + "_local"), i32)
            rank = IRBuilder.lshr(address, self.ir.Constant(i32, 24), tag + "_rank")
            mapped.extend((local, rank))
        value = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "stas_value"), i32, "stas_value_bits")
        asm_ty = self.ir.FunctionType(self.ir.VoidType(), [i32] * 5)
        asm = self.ir.InlineAsm(
            asm_ty,
            "{.reg .u32 d, b; mapa.shared::cluster.u32 d, $0, $1; "
            "mapa.shared::cluster.u32 b, $2, $3; "
            "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [d], $4, [b];}",
            "r,r,r,r,r,~{memory}", side_effect=True,
        )
        IRBuilder.call(asm, [*mapped, value])

    @lift_for("STSM")
    def lift_stsm(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """STSM.16.[M|MT]88.N -> PTX stmatrix.sync.aligned.xN.m8n8[.trans].

        Warp-collective 8x8 16-bit matrix store: each lane addresses one
        16-byte row, the row DATA gathers fragments from across the warp. On
        the nvvm path emit the literal stmatrix op and let the hardware do the
        collective (the x86 path needs a runtime permutation instead). The
        source quad arrives as one packed value (parser groups N regs); split
        it back into N b32 words for the brace list.
        """
        uses = Inst.get_uses()
        ptr, val_op = uses[0], uses[1]
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        count = int(Inst.opcodes[-1]) if Inst.opcodes[-1].isdigit() else 1
        trans = any("MT88" in m for m in Inst.opcodes)

        addr = _get_val(ptr, "stsm_addr")
        gep = self._shared_gep(IRBuilder, addr, "stsm_shared")
        addr64 = IRBuilder.ptrtoint(gep, i64, "stsm_ptr")

        v = _get_val(val_op, "stsm_val")
        total_ty = self.ir.IntType(count * 32)
        v = self._insert_type_cast_if_needed(IRBuilder, v, total_ty, "stsm_val_cast")
        words = []
        for k in range(count):
            w = v
            if count > 1:
                if k:
                    w = IRBuilder.lshr(w, self.ir.Constant(total_ty, 32 * k),
                                       f"stsm_w{k}_s")
                w = IRBuilder.trunc(w, i32, f"stsm_w{k}")
            words.append(w)

        t = ".trans" if trans else ""
        regs = "{" + ",".join(f"${i + 1}" for i in range(count)) + "}"
        asm_str = f"stmatrix.sync.aligned.x{count}.m8n8{t}.shared.b16 [$0], {regs};"
        cons = "l," + ",".join(["r"] * count) + ",~{memory}"
        fn_ty = self.ir.FunctionType(self.ir.VoidType(), [i64] + [i32] * count)
        asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
        IRBuilder.call(asm, [addr64] + words, "")

    @lift_for("STS")
    def lift_sts(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "sts_addr")
        v = _get_val(val, "sts_val")

        # Narrow-width modifiers: STS.U8/S8/U16/S16 — see lift_stg.
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        if "U8" in mods or "S8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "U16" in mods or "S16" in mods:
            narrow_ty = self.ir.IntType(16)
        if narrow_ty is not None and isinstance(v.type, self.ir.IntType) and v.type.width > narrow_ty.width:
            v = IRBuilder.trunc(v, narrow_ty, "sts_trunc")

        pointee_ty = v.type
        addr = self._shared_gep(IRBuilder, addr, "sts_shared_addr")
        if addr.type.pointee != pointee_ty:
            addr = IRBuilder.bitcast(
                addr,
                self.ir.PointerType(pointee_ty, 3),
                "sts_shared_addr_cast",
            )
        sts_st = IRBuilder.store(v, addr)
        # Volatile so llc -O2 can't dead-store-eliminate warp-synchronous shared
        # writes that cross-lane reads depend on (see _patch_volatile_memops).
        sts_st.volatile = True

    @lift_for("CALLEXT")
    def lift_callext(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """Extern device-runtime call resolved from the bank-4 pointer table
        (ResolveIndirectCall). The GPU device-call ABI passes the first two
        64-bit params in R4:R5 / R6:R7 and returns in R4.

        __assertfail is lowered to the PTX-native `__assertfail` external (the
        CUDA driver provides it during cuModuleLoad/ptxas link). Calling it sets
        the device-side assert flag so the host's cudaDeviceSynchronize returns
        cudaErrorAssert — matching the original SASS, where the failing thread
        CALLs the relocated __assertfail. Its declared signature mirrors what
        clang emits for `assert`: (const char* msg, const char* file, unsigned
        line, const char* func, size_t charSize). The message/file pointers come
        from unmaterialized bank-4 $str slots, so pass NULLs; the assert still
        fires regardless of message content (cudaErrorAssert is what the test
        observes). vprintf is declared `i32 @vprintf(i8*, i8*)` (host-resolved).
        """
        name = Inst.ext_callee if hasattr(Inst, "ext_callee") else Inst.opcodes[1]
        i8ptr = self.ir.PointerType(self.ir.IntType(8))
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        uses = Inst.get_uses()
        if name == "__assertfail":
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

    @lift_for("LDL")
    def lift_ldl(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        ptr = Inst.get_uses()[0]
        addr = _get_val(ptr, "ldl_addr")
        # SASS LDL/STL use byte offsets (R1 is the byte-addressed stack
        # pointer + literal byte offset); LocalMem is [N x i8], so the
        # gep index is the byte address directly.
        dest_ty = dest.get_ir_type(self)

        # Narrow-width modifiers: LDL.U8/S8/U16/S16 — see lift_ldg.
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        is_signed = False
        if "U8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "S8" in mods:
            narrow_ty = self.ir.IntType(8); is_signed = True
        elif "U16" in mods:
            narrow_ty = self.ir.IntType(16)
        elif "S16" in mods:
            narrow_ty = self.ir.IntType(16); is_signed = True
        use_narrow = narrow_ty is not None and isinstance(dest_ty, self.ir.IntType)
        pointee_ty = narrow_ty if use_narrow else dest_ty

        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "ldl_local_addr")
        addr = IRBuilder.bitcast(addr, self.ir.PointerType(pointee_ty), "ldl_local_addr_cast")
        val = IRBuilder.load(addr, "ldl", typ=pointee_ty)
        if use_narrow and dest_ty != narrow_ty:
            if is_signed:
                val = IRBuilder.sext(val, dest_ty, "ldl_sext")
            else:
                val = IRBuilder.zext(val, dest_ty, "ldl_zext")
        self._set_val(IRBuilder, IRRegs, dest, val)


    @lift_for("STL")
    def lift_stl(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        uses = Inst.get_uses()
        ptr, val = uses[0], uses[1]
        addr = _get_val(ptr, "stl_addr")
        v = _get_val(val, "stl_val")

        # Narrow-width modifiers: STL.U8/S8/U16/S16 — see lift_stg.
        mods = [tok.upper() for tok in Inst.opcodes[1:]]
        narrow_ty = None
        if "U8" in mods or "S8" in mods:
            narrow_ty = self.ir.IntType(8)
        elif "U16" in mods or "S16" in mods:
            narrow_ty = self.ir.IntType(16)
        if narrow_ty is not None and isinstance(v.type, self.ir.IntType) and v.type.width > narrow_ty.width:
            v = IRBuilder.trunc(v, narrow_ty, "stl_trunc")

        addr = IRBuilder.gep(self.LocalMem, [self.ir.Constant(self.ir.IntType(32), 0), addr], "stl_local_addr")
        addr = IRBuilder.bitcast(addr, self.ir.PointerType(v.type), "stl_local_addr_cast")
        IRBuilder.store(v, addr)


    @lift_for("CS2R")
    def lift_cs2r(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        ResOp = Inst.get_defs()[0]
        ValOp = Inst.get_uses()[0]
        if ResOp.is_reg or ResOp.is_reg_group:
            dest_name = ResOp.get_ir_name(self)

            def _call_sreg(sreg_name, tag):
                return self.run_intrinsic(f"sreg_{sreg_name}", IRBuilder, [], tag)

            # Un-suffixed `CS2R Rn, SR_*` is the 64-bit form: sir/instruction.py
            # auto-pairs Rn into the reg_group R(n+1):Rn. For SR_CLOCKLO and
            # SR_GLOBALTIMER_LO the SASS shorthand actually means the full
            # 64-bit timer (low gets Rn, high gets R(n+1)). For all other SRs
            # (TID/CTAID/laneid/...) the source is 32-bit and the high half is
            # implicitly zero — _set_val zext-then-splits handles that.
            is_pair = ResOp.is_reg_group and len(ResOp.reg_group) == 2

            if ValOp.is_rz:
                IRVal = self.ir.Constant(self.ir.IntType(32), 0)
            elif ValOp.is_clock_lo and is_pair:
                IRVal = self.run_intrinsic("sreg_clock64", IRBuilder, [], "cs2r_clock64")
            elif ValOp.is_global_timer_lo and is_pair:
                IRVal = self.run_intrinsic("sreg_globaltimer", IRBuilder, [], "cs2r_globaltimer64")
            elif ValOp.is_thread_idx_x or (ValOp.is_thread_idx and not ValOp.special_register_axis):
                IRVal = _call_sreg("tid.x", "cs2r_tid_x")
            elif ValOp.is_thread_idx_y:
                IRVal = _call_sreg("tid.y", "cs2r_tid_y")
            elif ValOp.is_thread_idx_z:
                IRVal = _call_sreg("tid.z", "cs2r_tid_z")
            elif ValOp.is_block_dim_x or (ValOp.is_block_dim and not ValOp.special_register_axis):
                IRVal = _call_sreg("ntid.x", "cs2r_ntid_x")
            elif ValOp.is_block_dim_y:
                IRVal = _call_sreg("ntid.y", "cs2r_ntid_y")
            elif ValOp.is_block_dim_z:
                IRVal = _call_sreg("ntid.z", "cs2r_ntid_z")
            elif ValOp.is_block_idx_x or (ValOp.is_block_idx and not ValOp.special_register_axis):
                IRVal = _call_sreg("ctaid.x", "cs2r_ctaid_x")
            elif ValOp.is_block_idx_y:
                IRVal = _call_sreg("ctaid.y", "cs2r_ctaid_y")
            elif ValOp.is_block_idx_z:
                IRVal = _call_sreg("ctaid.z", "cs2r_ctaid_z")
            elif ValOp.is_grid_dim_x or (ValOp.is_grid_dim and not ValOp.special_register_axis):
                IRVal = _call_sreg("nctaid.x", "cs2r_nctaid_x")
            elif ValOp.is_grid_dim_y:
                IRVal = _call_sreg("nctaid.y", "cs2r_nctaid_y")
            elif ValOp.is_grid_dim_z:
                IRVal = _call_sreg("nctaid.z", "cs2r_nctaid_z")
            elif ValOp.is_lane_id:
                IRVal = _call_sreg("laneid", "cs2r_lane")
            elif ValOp.is_warp_id:
                IRVal = _call_sreg("warpid", "cs2r_warp")
            elif ValOp.is_warp_size:
                IRVal = _call_sreg("warpsize", "cs2r_warpsize")
            elif ValOp.is_active_mask:
                # See lift_s2r: activemask is not a %sreg.
                IRVal = self.run_intrinsic("activemask", IRBuilder, [], "cs2r_activemask")
            elif ValOp.is_lane_mask_eq:
                IRVal = _call_sreg("lanemask.eq", "cs2r_lanemask_eq")
            elif ValOp.is_lane_mask_le:
                IRVal = _call_sreg("lanemask.le", "cs2r_lanemask_le")
            elif ValOp.is_lane_mask_lt:
                IRVal = _call_sreg("lanemask.lt", "cs2r_lanemask_lt")
            elif ValOp.is_lane_mask_ge:
                IRVal = _call_sreg("lanemask.ge", "cs2r_lanemask_ge")
            elif ValOp.is_lane_mask_gt:
                IRVal = _call_sreg("lanemask.gt", "cs2r_lanemask_gt")
            elif ValOp.is_clock_lo:
                IRVal = _call_sreg("clock", "cs2r_clock_lo")
            elif ValOp.is_clock_hi:
                clk64 = self.run_intrinsic("sreg_clock64", IRBuilder, [], "cs2r_clock64_for_hi")
                shifted = IRBuilder.lshr(clk64, self.ir.Constant(self.ir.IntType(64), 32), "cs2r_clock_hi_shr")
                IRVal = IRBuilder.trunc(shifted, self.ir.IntType(32), "cs2r_clock_hi")
            elif ValOp.is_global_timer_lo:
                IRVal = _call_sreg("globaltimer.lo", "cs2r_gtimer_lo")
            elif ValOp.is_global_timer_hi:
                IRVal = _call_sreg("globaltimer.hi", "cs2r_gtimer_hi")
            else:
                print(f"CS2R: Unknown special register {ValOp}")
                IRVal = self.ir.Constant(self.ir.IntType(32), 0)

            if is_pair:
                # _set_val on a reg_group casts/zext-splits IRVal across the
                # pair: low = trunc(val, i32), high = trunc(val>>32, i32).
                # 32-bit SRs (TID/CTAID/...) get zext'd, so high lands as 0.
                self._set_val(IRBuilder, IRRegs, ResOp, IRVal)
            elif dest_name in IRRegs and IRRegs[dest_name].type.is_pointer:
                IRBuilder.store(IRVal, IRRegs[dest_name])
            else:
                IRRegs[dest_name] = IRVal


    @lift_for("VOTE", "VOTEU")
    def lift_vote(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        if len(uses) >= 2:
            mask_op, pred_op = uses[0], uses[1]
        else:
            mask_op = None
            pred_op = uses[0]

        pred_val = _get_val(pred_op, "vote_pred")

        if mask_op is None or (mask_op.is_pt and not mask_op.is_not_reg):
            mask_val = self.ir.Constant(self.ir.IntType(32), -1)
        else:
            mask_val = _get_val(mask_op, "vote_mask")

        # SASS `VOTE.<mode> Rd, Pd, Ps` produces TWO results: the ballot mask
        # (which lanes have Ps set) into the *register* Rd, and the .ANY/.ALL/.UNI
        # reduction into the *predicate* Pd. The mode selects only the predicate
        # reduction — a register destination always receives the ballot. Emitting
        # vote.any (a boolean) into a register dest drops the per-lane mask, so
        # any popc(ballot)-based counting (e.g. nvtiff LZW output positioning)
        # comes out wrong.
        is_pred_dest = dest.is_pt or dest.is_predicate_reg

        # `VOTE.ANY Rd, PT, PT` (register ballot of constant-true) is exactly
        # __activemask(): the mask of currently-active lanes. Lower it to PTX
        # `activemask.b32`, NOT vote.sync.ballot(true, -1) — vote.sync with
        # membermask -1 asserts all 32 lanes are converged, which is undefined
        # in the divergent regions these reads sit in (lda-cuda's ReduceSum
        # calls __activemask() under `if (warp == 0)`), and the resulting
        # branch-over-vote.sync PTX SIGSEGVs ptxas 12.2 at -O1+ (valid PTX;
        # -O0 accepts in the reduced cluster reproducer).
        if (not is_pred_dest
                and pred_op.is_pt and not pred_op.is_not_reg
                and (mask_op is None
                     or (mask_op.is_pt and not mask_op.is_not_reg))):
            vote_val = self.run_intrinsic("activemask", IRBuilder, [], "activemask")
        else:
            if is_pred_dest:
                mode = Inst.opcodes[1].lower()
                intrinsic_name = f"vote_{mode}"
            else:
                intrinsic_name = "vote_ballot"
            vote_val = self.run_intrinsic(intrinsic_name, IRBuilder, [mask_val, pred_val], intrinsic_name)

        self._set_val(IRBuilder, IRRegs, dest, vote_val)

    def _shfl_membermask(self, Inst, mode, IRBuilder, IRRegs, _get_val):
        """Context-aware membermask for `shfl.sync` on the NVVM backend.

        Two distinct SHFL classes exist, and they need DIFFERENT membermasks:

        1. Converged warp scans (`SHFL.UP/DOWN/BFLY`, e.g. cub BlockScan /
           segsort) run with the HARDWARE warp fully converged. The correct
           membermask is -1 (all 32 lanes). A partial mask here gives wrong
           scan results, so these MUST keep -1.

        2. Warp-aggregated atomics / `coalesced_threads()` (`SHFL.IDX`) run
           under a PARTIAL-warp predicate — only the active lanes participate.
           The SASS idiom is:
               VOTEU.ANY URm, UPT, PT      ; URm = active-lane mask
               FLO.U32   Rlane, URm        ; Rlane = leader lane id
               SHFL.IDX  PT, Rd, Rsrc, Rlane, 0x1f
           A membermask of -1 here makes `shfl.sync` block on the 31 absent
           lanes -> CTA deadlock at the next __syncthreads (threadfence /
           filter timeout cluster).

        Discriminator: `mode == "idx"` selects class 2. For those we recover
        the idiom's OWN active mask by tracing the source-lane (FLO/POPC-style)
        operand back through its single reaching def to the `VOTE`/`VOTEU`
        instruction that produced the mask, and reuse that register's already-
        computed IR value (the same `activemask` value FLO consumed). This is
        exactly the hand-edit that fixes the deadlock, and — unlike a FRESH
        `activemask` read — it is read at the SAME convergence point the
        VOTEU was, so it returns the kernel's true active mask. If the trace
        fails (no recoverable VOTE source), fall back to a fresh `activemask`
        read, which is still correct for the consecutive-block IDX idiom and
        strictly safer than -1 for a partial warp.

        Non-IDX modes keep -1 (class 1), so converged scans are untouched.
        """
        if mode != "idx":
            return self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)

        mask_op = self._trace_shfl_active_mask(Inst)
        if mask_op is not None:
            return self._get_val(IRBuilder, IRRegs, mask_op, "shfl_membermask")

        # No recoverable VOTE source: a partial-warp IDX still must not use -1.
        return self.run_intrinsic("activemask", IRBuilder, [], "shfl_activemask")

    def _trace_shfl_active_mask(self, Inst):
        """Trace `SHFL.IDX`'s source-lane operand back to the VOTE/VOTEU mask
        operand that produced its active-lane mask, returning that mask
        operand (or None if the idiom isn't matched).

        Chain: SHFL.offset --reaching-def--> FLO.U32 --reaching-def--> VOTE/VOTEU.
        Uses the def-use info populated by the DefUseAnalysis that runs just
        before lifting (operands carry `reaching_defs`).
        """
        uses = Inst.get_uses()
        if len(uses) < 2:
            return None
        lane_op = uses[1]

        reaching = getattr(Inst, "reaching_defs", None)
        if not reaching:
            return None
        flo_inst = reaching.get(lane_op)
        # Walk back through single-source pass-through ops (FLO/POPC/MOV) until
        # we hit a VOTE/VOTEU that produced the active mask.
        visited = set()
        cur = flo_inst
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))
            op0 = cur.opcodes[0] if cur.opcodes else ""
            if op0 in ("VOTE", "VOTEU"):
                defs = cur.get_defs()
                return defs[0] if defs else None
            if op0 in ("FLO", "POPC", "UPOPC", "MOV", "UMOV"):
                cur_uses = cur.get_uses()
                if not cur_uses:
                    return None
                cur = getattr(cur, "reaching_defs", {}).get(cur_uses[0])
                continue
            return None
        return None


    @lift_for("DMMA")
    def lift_dmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """DMMA on NVVM -> native f64 tensor-core MMA intrinsic.

        The base Lifter.lift_dmma emits the x86 warp-gather wrapper
        (dmma_884_f64 / dmma_16816_f64) — a host-side helper that resolves
        in the CPU driver but becomes an UNRESOLVED EXTERN in PTX
        (ptxas fatal: Unresolved extern function 'dmma_884_f64'), failing
        the JIT for every f64 CUTLASS gemm (18_ampere_fp64). On real GPU the
        tensor core does the warp-collective natively, so emit the
        llvm.nvvm.mma.*.f64 intrinsic (mirrors how HMMA routes through
        run_intrinsic)."""
        shape = Inst.opcodes[1] if len(Inst.opcodes) > 1 else ""
        uses = Inst.get_uses()
        dests = Inst.get_defs()
        f64 = self.ir.DoubleType()

        if shape == "8x8x4":
            iname, n_use, n_def = "dmma884f64", 4, 2
        elif shape == "16x8x16":
            iname, n_use, n_def = "dmma16816f64", 16, 4
        else:
            raise UnsupportedInstructionException(f"Unsupported DMMA shape: {shape}")

        if len(uses) != n_use or len(dests) != n_def:
            raise UnsupportedInstructionException(
                f"DMMA.{shape} expects {n_use} uses / {n_def} defs, "
                f"got {len(uses)}/{len(dests)}")

        args = []
        for i, u in enumerate(uses):
            v = _get_val(u, f"dmma_arg_{i}")
            if v.type != f64:
                v = self._insert_type_cast_if_needed(IRBuilder, v, f64, f"dmma_arg_{i}")
            args.append(v)

        val = self.run_intrinsic(iname, IRBuilder, args, "dmma_call")
        for i in range(len(dests)):
            self._set_val(IRBuilder, IRRegs, dests[i],
                          IRBuilder.extract_value(val, i, f"dmma_d{i}"))

    @lift_for("QMMA")
    def lift_qmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # ptxas uses QMMA D,A,B,C,UPn when a uniform select zeros C if UPn
        # is false. Preserve that select before the warp-level MMA.
        mods = Inst.opcodes[1:]
        uses = Inst.get_uses()
        if (len(mods) != 4 or mods[:2] != ["16832", "F32"] or
                any(t not in ("E4M3", "E5M2") for t in mods[2:]) or
                len(uses) not in (10, 11) or len(Inst.get_defs()) != 4):
            raise UnsupportedInstructionException(f"Unsupported QMMA form: {Inst.opcodes}")
        i32, f32 = self.ir.IntType(32), self.ir.FloatType()
        types = [i32] * 6 + [f32] * 4
        values = [self._insert_type_cast_if_needed(IRBuilder, _get_val(op, "qmma_input"), ty, "qmma_input")
                  for op, ty in zip(uses, types)]
        if len(uses) == 11:
            predicate = _get_val(uses[-1], "qmma_use_c")
            for j in range(6, 10):
                values[j] = IRBuilder.select(predicate, values[j], self.ir.Constant(f32, 0.0))
        result_type = self.ir.LiteralStructType([f32] * 4)
        asm = self.ir.InlineAsm(
            self.ir.FunctionType(result_type, types),
            f"mma.sync.aligned.m16n8k32.row.col.f32.{mods[2].lower()}.{mods[3].lower()}.f32 "
            "{$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", side_effect=False)
        result = IRBuilder.call(asm, values, "qmma")
        for j, dest in enumerate(Inst.get_defs()):
            self._set_val(IRBuilder, IRRegs, dest, IRBuilder.extract_value(result, j))

    @lift_for("IMMA")
    def lift_imma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """IMMA on NVVM -> native integer tensor-core MMA intrinsic.

        The base Lifter.lift_imma routes the common dense shapes (8816/16832
        s8) and every sparse shape through x86 warp-gather wrappers
        (imma_8816_s8_warp, imma_sp16864_s8_warp, ...) — host helpers that
        resolve in the CPU driver but are UNRESOLVED EXTERNS in PTX
        (ptxas fatal: Unresolved extern function 'imma_8816_s8_warp'),
        failing the JIT for the int8 CUTLASS gemms (08_turing, 09_turing).
        On GPU the tensor core is native, so emit
        llvm.nvvm.mma.<geom>.row.col[.satfinite].<ty> (the run_intrinsic
        imma_ path derives the fragment counts from the geometry)."""
        is_sparse = len(Inst.opcodes) > 1 and Inst.opcodes[1] == "SP"
        base = 2 if is_sparse else 1
        shape = Inst.opcodes[base]
        src_type_a = Inst.opcodes[base + 1] if len(Inst.opcodes) > base + 1 else ""
        src_type_b = Inst.opcodes[base + 2] if len(Inst.opcodes) > base + 2 else ""
        sat = "SAT" in Inst.opcodes

        uses = Inst.get_uses()
        dests = Inst.get_defs()
        i32 = self.ir.IntType(32)

        shape_map = {
            "8816": "m8n8k16", "8832": "m8n8k32",
            "16816": "m16n8k16", "16832": "m16n8k32", "16864": "m16n8k64",
        }
        if is_sparse:
            # mma.sp (2:4 structured sparse) has no stable named NVVM
            # intrinsic across LLVM versions. Leave a placeholder rather than
            # emit an unresolved extern / invalid PTX (15_ampere_sparse stays
            # failing on correctness, but the JIT no longer crashes).
            raise UnsupportedInstructionException(
                f"Sparse IMMA.SP.{shape} not supported on NVVM backend yet")
        if shape not in shape_map:
            raise UnsupportedInstructionException(f"Unsupported IMMA shape: {shape}")

        args = [
            self._insert_type_cast_if_needed(
                IRBuilder, _get_val(uses[i], f"imma_arg_{i}"), i32, f"imma_arg_{i}")
            for i in range(len(uses))
        ]

        a_ty = src_type_a.lower()
        b_ty = src_type_b.lower()
        sat_str = "_satfinite" if sat else ""
        type_suffix = f"_{a_ty}" if a_ty == b_ty else f"_{a_ty}_{b_ty}"
        intrinsic_name = f"imma_{shape_map[shape]}{sat_str}{type_suffix}"
        val = self.run_intrinsic(intrinsic_name, IRBuilder, args, "imma_call")
        for i in range(len(dests)):
            self._set_val(IRBuilder, IRRegs, dests[i],
                          IRBuilder.extract_value(val, i, f"imma_res_{i}"))

    @lift_for("WARPRECONV")
    def lift_warpreconv(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """Synthetic warp-reconvergence (WarpReconverge pass) -> bar.warp.sync
        over the ACTIVE-lane mask. Reconverges the warp at the merge of a
        branch-split predicated shared-memory region so warp-synchronous
        reductions see all lanes' shared writes. Uses activemask() (the
        __syncwarp() idiom) rather than a full 0xffffffff mask: a full mask
        DEADLOCKS when not every lane reaches the merge (segsort timeout); the
        active mask only waits for lanes actually present, so it is deadlock-safe
        while still reconverging the lanes that participate in the reduction."""
        mask = self.run_intrinsic("activemask", IRBuilder, [], "warp_reconv_mask")
        self.run_intrinsic("bar_warp_sync", IRBuilder, [mask], "warp_reconv")

    @lift_for("UCGABAR_ARV", "UCGABAR_WAIT")
    def lift_cluster_barrier(self, Inst, IRBuilder, IRRegs, block_map=None, **kwargs):
        operation = "arrive" if Inst.opcodes[0] == "UCGABAR_ARV" else "wait"
        asm = self.ir.InlineAsm(
            self.ir.FunctionType(self.ir.VoidType(), []),
            f"barrier.cluster.{operation}.aligned;", "~{memory}", side_effect=True,
        )
        IRBuilder.call(asm, [])

    @lift_for("MEMBAR")
    def lift_membar(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """MEMBAR -> real NVVM memory fence (base Lifter NOPs it).

        __threadfence{,_block,_system}() compiles to MEMBAR; the base
        lift_membar drops it, which is fine for kernels that only use
        atomics for relaxed counters but BREAKS lock-free structures whose
        correctness depends on the fence ordering. btree (UC-Davis GpuBTree)
        release_lock does `__threadfence(); atomicAnd(unlock)` — without the
        fence the unlock floats above the node-data writes, a concurrent warp
        acquires the lock and reads a half-written node, traversal corrupts,
        node splits run away -> atomicAdd allocation counter explodes
        (212 GiB tree / illegal memory access vs 0.628 GiB correct).
        Scope: .CTA -> membar.cta, .SYS -> membar.sys, else (.GPU/.SC.GPU)
        -> membar.gl. Emitted only where the original SASS has a MEMBAR, so
        kernels without fences are unaffected (no perf regression)."""
        mods = [t.upper() for t in Inst.opcodes[1:]]
        if "CTA" in mods:
            scope = "membar_cta"
        elif "SYS" in mods:
            scope = "membar_sys"
        else:
            scope = "membar_gl"
        self.run_intrinsic(scope, IRBuilder, [], scope)

    @lift_for("FENCE")
    def lift_fence(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """SM90 FENCE.VIEW.ASYNC -> fence.proxy.async (async-proxy fence).

        Hopper's mbarrier/TMA hardware reads through the ASYNC proxy; writes
        made through the generic proxy (e.g. the SYNCS.EXCH that installs the
        encoded mbarrier init state) are not guaranteed visible to it until a
        proxy fence. Dropping the fence leaves the mbarrier uninitialised from
        the async unit's view -> arrive/try_wait fault ("unspecified launch
        failure") — this is required-for-correctness, not a hint.
          FENCE.VIEW.ASYNC.S -> fence.proxy.async.shared::cta;
          FENCE.VIEW.ASYNC.G -> fence.proxy.async.global;
          other FENCE forms  -> fence.acq_rel scoped like MEMBAR (conservative).
        """
        mods = [t.upper() for t in Inst.opcodes[1:]]
        if "ASYNC" in mods or "VIEW" in mods:
            space = "shared::cta" if "S" in mods else (
                "global" if "G" in mods else "shared::cta")
            asm_ty = self.ir.FunctionType(self.ir.VoidType(), [])
            asm = self.ir.InlineAsm(
                asm_ty, f"fence.proxy.async.{space};", "~{memory}", side_effect=True)
            IRBuilder.call(asm, [], "")
            return
        # Non-async FENCE: conservative acq_rel fence at CTA scope.
        self.run_intrinsic("membar_cta", IRBuilder, [], "membar_cta")

    @lift_for("NANOSLEEP")
    def lift_nanosleep(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # NANOSLEEP.SYNCS t — spin-loop backoff hint (decoupled-lookback scan,
        # spin-locks). Semantically a NOP (busy-wait is still correct). It was
        # falling through to an opaque placeholder, which becomes a real call in
        # the spin loop -> ~60x slowdown / apparent hang (scan3, pad TIMEOUT).
        # NOP removes that overhead. (CPU backend already NOPs it.)
        pass

    @lift_for("REDUX")
    def lift_redux(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # REDUX.<OP>[.S32] Rd, Rsrc — warp-wide reduction; EVERY active lane
        # receives the result. The base Lifter identity-lifts it (correct only
        # for the x86 warp-of-1 replay); on a real GPU warp identity DROPS
        # every other lane's contribution. nvcc's atomic-aggregation idiom
        # (REDUX.SUM ; VOTEU/FLO leader-elect ; @leader REDG) then writes only
        # the leader lane's own value -> wrong sums. Lower to the hardware
        # warp-reduce intrinsic redux.sync.<op> (sm_80+), full-warp membermask.
        dest = Inst.get_defs()[0]
        val = _get_val(Inst.get_uses()[0], "redux_src")
        mods = [t.upper() for t in Inst.opcodes[1:]]
        op = next((m for m in mods if m in ("SUM", "ADD", "IADD", "MIN", "MAX", "AND", "OR", "XOR")), None)
        if op is None:
            # Bare "REDUX Rd, Rsrc" with no op modifier = bitwise AND. The REDUX
            # sub-op field encodes AND as its default value, so cuobjdump omits
            # the suffix (verified: `__reduce_and_sync` disassembles to bare
            # `REDUX`, while `__reduce_add_sync` is `REDUX.SUM`). nvcc's
            # warp-aggregated system/global AND atomic uses this bare form to
            # pre-reduce the source across the warp, then a leader-lane REDG.AND
            # commits it (atomicSystemWide/atomicIntrinsics/btree all feed the
            # bare REDUX result into REDG.E.AND). Lifting it as ADD produced a
            # wrong AND operand (atomicSystemWide: "atomicAnd failed").
            op = "AND"
        is_signed = "S32" in mods
        iname = {"SUM": "redux_sync_add", "ADD": "redux_sync_add", "IADD": "redux_sync_add",
                 "MIN": "redux_sync_min" if is_signed else "redux_sync_umin",
                 "MAX": "redux_sync_max" if is_signed else "redux_sync_umax",
                 "AND": "redux_sync_and", "OR": "redux_sync_or", "XOR": "redux_sync_xor"}[op]
        i32 = self.ir.IntType(32)
        if not (isinstance(val.type, self.ir.IntType) and val.type.width == 32):
            val = (IRBuilder.trunc(val, i32, "redux_src_i32")
                   if isinstance(val.type, self.ir.IntType) and val.type.width > 32
                   else IRBuilder.bitcast(val, i32, "redux_src_i32"))
        # Membermask = the ACTIVE lanes, not a hardcoded full warp. HW REDUX.SUM
        # reduces only the currently-active lanes; in a divergent context (e.g.
        # nvcc's warp-aggregated atomicAdd: VOTEU ballot -> REDUX.SUM -> @leader
        # REDG) a full-warp 0xFFFFFFFF mask folds in inactive lanes' stale values
        # -> over-count. activemask() matches the real participating set.
        mask = self.run_intrinsic("activemask", IRBuilder, [], "redux_mask")
        res = self.run_intrinsic(iname, IRBuilder, [val, mask], iname)
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("REDG")
    def lift_redg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        # SM90 global reduction: REDG.E.<OP>.<TYPE> desc[URx][Ry.64], Rsrc
        # (fire-and-forget atomic to global / addrspace 1). The NVVM backend
        # had NO REDG handler — only RED — so every REDG was emitted as a no-op
        # placeholder and the reduction was silently DROPPED, leaving the
        # destination at its initial value (zeros / wrong results). Mirrors
        # lift_red but covers the full type/op matrix (F64, 64-bit ints,
        # INC/DEC) and uses the ATOMG descriptor-address resolution path.
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        op_type = None
        for mod in modifiers:
            if mod in ("ADD", "MIN", "MAX", "AND", "OR", "XOR", "INC", "DEC"):
                op_type = mod
                break
        if op_type is None:
            raise UnsupportedInstructionException(f"Unknown REDG operation: {Inst.opcodes}")

        uses = Inst.get_uses()
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"No memory operand in REDG: {Inst.opcodes}")
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "redg_addr")

        # REDG.E.ADD.F32x{2,4}: vector float reduce — N consecutive f32 adds
        # to consecutive gmem words (FA3 bwd hdim256 dQaccum). The parser
        # groups the N source regs into one packed value. Without this branch
        # the exact-match "F32" test below misses "F32X4" and the packed word
        # fell into the INTEGER add path: one lane of f32 bits int-added,
        # three lanes dropped — dQ came back inf over the unwritten half.
        vec_w = 4 if "F32X4" in modifiers else (2 if "F32X2" in modifiers else 1)
        if vec_w > 1 and op_type == "ADD":
            f32 = self.ir.FloatType()
            i32 = self.ir.IntType(32)
            total_ty = self.ir.IntType(32 * vec_w)
            v = _get_val(uses[-1], "redg_vec")
            v = self._insert_type_cast_if_needed(IRBuilder, v, total_ty, "redg_vec_cast")
            base = _as_pointer(addr_val, f32, "redg_vptr", addrspace=1)
            for k in range(vec_w):
                w = v
                if k:
                    w = IRBuilder.lshr(w, self.ir.Constant(total_ty, 32 * k), f"redg_v{k}_s")
                w = IRBuilder.trunc(w, i32, f"redg_v{k}_w")
                wf = IRBuilder.bitcast(w, f32, f"redg_v{k}_f")
                p = IRBuilder.gep(base, [self.ir.Constant(i32, k)], name=f"redg_v{k}_p")
                IRBuilder.atomic_rmw("fadd", p, wf, "monotonic", f"redg_v{k}_fadd")
            return

        is_packed_half = "F16X2" in modifiers or "BF16X2" in modifiers
        is_bf = "BF16X2" in modifiers or "BF16" in modifiers
        is_64 = "64" in modifiers or "S64" in modifiers or "U64" in modifiers
        is_signed = "S32" in modifiers or "S64" in modifiers
        is_float_add = op_type == "ADD" and (
            is_packed_half or "F32" in modifiers or "F64" in modifiers
            or "F16" in modifiers or "BF16" in modifiers)

        if is_packed_half:
            elem_ty = self.ir.IntType(32)          # _atom_add_f16x2 operates on the i32 word
        elif "F64" in modifiers:
            elem_ty = self.ir.DoubleType()
        elif "F32" in modifiers:
            elem_ty = self.ir.FloatType()
        elif "F16" in modifiers or "BF16" in modifiers:
            elem_ty = self._half2_elem_ty(is_bf)
        else:
            elem_ty = self.ir.IntType(64) if is_64 else self.ir.IntType(32)

        ptr = _as_pointer(addr_val, elem_ty, "redg_ptr", addrspace=1)
        src_val = _get_val(uses[-1], "redg_val")
        src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "redg_val")

        if is_packed_half and op_type == "ADD":
            self._atom_add_f16x2(IRBuilder, ptr, src_val)
        elif is_float_add:
            IRBuilder.atomic_rmw("fadd", ptr, src_val, "monotonic", "redg_fadd")
        elif op_type == "INC":
            # NVIDIA INC: new = (old >= val) ? 0 : old + 1  (no return value).
            # MUST be a single atomic RMW — a load/select/store sequence races
            # under grid-wide contention (atomicIntrinsics' 1<<27 threads all
            # INC the same word -> nondeterministic garbage). LLVM uinc_wrap has
            # exactly this semantics; mirrors lift_atomg's INC handling.
            IRBuilder.atomic_rmw("uinc_wrap", ptr, src_val, "monotonic", "redg_inc")
        elif op_type == "DEC":
            # NVIDIA DEC: new = (old == 0 || old > val) ? val : old - 1.
            # Single atomic RMW (LLVM udec_wrap) for the same race reason as INC.
            IRBuilder.atomic_rmw("udec_wrap", ptr, src_val, "monotonic", "redg_dec")
        elif "SYS" in modifiers and self._atom_sys_intrinsic(
                IRBuilder, ptr, op_type, src_val, name=f"redg_sys_{op_type.lower()}") is not None:
            pass  # system-scope int atomic emitted via nvvm .sys intrinsic
        else:
            rmw = {"ADD": "add",
                   "MIN": "min" if is_signed else "umin",
                   "MAX": "max" if is_signed else "umax",
                   "AND": "and", "OR": "or", "XOR": "xor"}[op_type]
            IRBuilder.atomic_rmw(rmw, ptr, src_val, "monotonic", f"redg_{op_type.lower()}")

    @lift_for("RED")
    def lift_red(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        op_type = None
        data_type = None
        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR"]:
                op_type = mod
            elif mod.startswith("F16X2") or mod == "F16X2":
                data_type = "F16x2"
            elif mod.startswith("F32") or mod == "F":
                data_type = "F32"
            elif mod.startswith("S32"):
                data_type = "S32"
            elif mod.startswith("U32"):
                data_type = "U32"
            elif mod.startswith("S64"):
                data_type = "S64"
            elif mod.startswith("U64"):
                data_type = "U64"

        if op_type is None:
            return

        uses = Inst.get_uses()
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            return
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "red_addr")
        src_op = uses[-1]
        src_val = _get_val(src_op, "red_val")

        if data_type == "F16x2":
            elem_ty = self.ir.IntType(32)
        elif data_type == "F32":
            elem_ty = self.ir.FloatType()
        elif data_type in ("S64", "U64"):
            elem_ty = self.ir.IntType(64)
        else:
            elem_ty = self.ir.IntType(32)

        ptr = _as_pointer(addr_val, elem_ty, "red_ptr", addrspace=1)
        src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "red_val")

        if data_type == "F16x2" and op_type == "ADD":
            self._atom_add_f16x2(IRBuilder, ptr, src_val)
        elif op_type == "ADD":
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                IRBuilder.atomic_rmw("fadd", ptr, src_val, "monotonic", "red_fadd")
            else:
                IRBuilder.atomic_rmw("add", ptr, src_val, "monotonic", "red_add")
        elif op_type == "MIN":
            op = "min" if data_type and data_type.startswith("S") else "umin"
            IRBuilder.atomic_rmw(op, ptr, src_val, "monotonic", "red_min")
        elif op_type == "MAX":
            op = "max" if data_type and data_type.startswith("S") else "umax"
            IRBuilder.atomic_rmw(op, ptr, src_val, "monotonic", "red_max")
        elif op_type == "AND":
            IRBuilder.atomic_rmw("and", ptr, src_val, "monotonic", "red_and")
        elif op_type == "OR":
            IRBuilder.atomic_rmw("or", ptr, src_val, "monotonic", "red_or")
        elif op_type == "XOR":
            IRBuilder.atomic_rmw("xor", ptr, src_val, "monotonic", "red_xor")

    @lift_for("ATOM")
    def lift_atom(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]

        op_type = None
        data_type = None
        is_signed = False

        for mod in modifiers:
            if mod in ["ADD", "MIN", "MAX", "AND", "OR", "XOR", "EXCH", "INC", "DEC", "CAS"]:
                op_type = mod
            elif mod.startswith("F16X2") or mod == "F16x2":
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
        
        dest_reg = defs[1]
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"No memory operand found in ATOM: {Inst.opcodes}")
        ptr_op = mem_ops[0]
        addr_val = _get_val(ptr_op, "atom_addr")
        
        src_op = uses[-1]
        src_val = _get_val(src_op, "atom_src_val")
        
        if data_type == "F16x2":
            # F16x2: two FP16 values packed in 32-bit word
            # Use i32 for the atomic operation, will be interpreted as packed FP16x2
            elem_ty = self.ir.IntType(32)
            rmw_op = "add"  # Will use intrinsic for FP add
            is_fp_add = (op_type == "ADD")
        elif data_type == "F32" or data_type is None:
            elem_ty = self.ir.FloatType()
            rmw_op = "fadd" if op_type == "ADD" else op_type.lower()
            is_fp_add = False
        elif data_type == "F64":
            elem_ty = self.ir.DoubleType()
            rmw_op = "fadd" if op_type == "ADD" else op_type.lower()
            is_fp_add = False
        elif data_type in ["S32", "U32"]:
            elem_ty = self.ir.IntType(32)
            is_fp_add = False
        elif data_type in ["S64", "U64"]:
            elem_ty = self.ir.IntType(64)
            is_fp_add = False
        else:
            elem_ty = self.ir.IntType(32)
            is_fp_add = False
        
        ptr = _as_pointer(
            addr_val,
            elem_ty,
            f"{ptr_op.get_ir_name(self)}_atom_addr" if ptr_op.is_reg else "atom_addr_ptr",
            addrspace=1,
        )
        
        if src_val.type != elem_ty:
            if isinstance(elem_ty, ir.IntType) and isinstance(src_val.type, ir.IntType):
                if elem_ty.width > src_val.type.width:
                    src_val = IRBuilder.zext(src_val, elem_ty, "atom_src_zext")
                elif elem_ty.width < src_val.type.width:
                    src_val = IRBuilder.trunc(src_val, elem_ty, "atom_src_trunc")
            else:
                src_val = IRBuilder.bitcast(src_val, elem_ty, "atom_src_cast")
        
        # System-scope (.SYS) integer atomics need cross-device coherence —
        # emit the nvvm `.sys` intrinsic (atom.sys.*) instead of a device-scope
        # atomicrmw. CAS is handled in its own branch below (two value operands).
        sys_result = None
        if ("SYS" in modifiers and op_type != "CAS"
                and isinstance(elem_ty, ir.IntType)):
            sys_result = self._atom_sys_intrinsic(
                IRBuilder, ptr, op_type, src_val, name=f"atom_sys_{op_type.lower()}")

        if sys_result is not None:
            result = sys_result
        elif data_type == "F16x2" and op_type == "ADD":
            result = self._atom_add_f16x2(IRBuilder, ptr, src_val)
        elif op_type == "ADD":
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                result = IRBuilder.atomic_rmw("fadd", ptr, src_val, "monotonic", "atom_fadd")
            else:
                result = IRBuilder.atomic_rmw("add", ptr, src_val, "monotonic", "atom_add")
        elif op_type == "MIN":
            if is_signed:
                result = IRBuilder.atomic_rmw("min", ptr, src_val, "monotonic", "atom_min")
            else:
                result = IRBuilder.atomic_rmw("umin", ptr, src_val, "monotonic", "atom_umin")
        elif op_type == "MAX":
            if is_signed:
                result = IRBuilder.atomic_rmw("max", ptr, src_val, "monotonic", "atom_max")
            else:
                result = IRBuilder.atomic_rmw("umax", ptr, src_val, "monotonic", "atom_umax")
        elif op_type == "AND":
            result = IRBuilder.atomic_rmw("and", ptr, src_val, "monotonic", "atom_and")
        elif op_type == "OR":
            result = IRBuilder.atomic_rmw("or", ptr, src_val, "monotonic", "atom_or")
        elif op_type == "XOR":
            result = IRBuilder.atomic_rmw("xor", ptr, src_val, "monotonic", "atom_xor")
        elif op_type == "EXCH":
            result = IRBuilder.atomic_rmw("xchg", ptr, src_val, "monotonic", "atom_exch")
        elif op_type == "CAS":
            cmp_val = _get_val(uses[-2], "atom_cmp")
            new_val = _get_val(uses[-1], "atom_new")
            cas_ty = elem_ty
            if isinstance(elem_ty, (ir.FloatType, ir.DoubleType)):
                bits = 32 if isinstance(elem_ty, ir.FloatType) else 64
                cas_ty = self.ir.IntType(bits)
                ptr = _as_pointer(addr_val, cas_ty, "atom_cas_ptr", addrspace=1)
            cmp_val = self._insert_type_cast_if_needed(IRBuilder, cmp_val, cas_ty, "atom_cmp")
            new_val = self._insert_type_cast_if_needed(IRBuilder, new_val, cas_ty, "atom_new")
            sys_cas = None
            if "SYS" in modifiers and isinstance(cas_ty, ir.IntType):
                sys_cas = self._atom_sys_intrinsic(
                    IRBuilder, ptr, "CAS", new_val, cmp_val=cmp_val, name="atom_sys_cas")
            if sys_cas is not None:
                result = sys_cas
            else:
                cas_result = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "monotonic", "monotonic", "atom_cas")
                result = IRBuilder.extract_value(cas_result, 0, "atom_cas_old")
            if cas_ty != elem_ty:
                result = IRBuilder.bitcast(result, elem_ty, "atom_cas_old_fp")
        else:
            raise UnsupportedInstructionException(f"Unsupported ATOM operation: {op_type}")
        
        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            dest_ty = dest_reg.get_ir_type(self)
            final_val = result
            if final_val.type != dest_ty:
                if isinstance(dest_ty, ir.IntType) and isinstance(final_val.type, ir.IntType):
                    if dest_ty.width > final_val.type.width:
                        final_val = IRBuilder.zext(final_val, dest_ty, "atom_result_zext")
                    elif dest_ty.width < final_val.type.width:
                        final_val = IRBuilder.trunc(final_val, dest_ty, "atom_result_trunc")
                elif (
                    isinstance(dest_ty, ir.IntType)
                    and isinstance(final_val.type, (ir.FloatType, ir.DoubleType))
                ):
                    final_val = IRBuilder.bitcast(result, dest_ty, "atom_result_bitcast")
                else:
                    final_val = IRBuilder.bitcast(result, dest_ty, "atom_result_cast")
            self._set_val(IRBuilder, IRRegs, dest_reg, final_val)
        
        defs = Inst.get_defs()
        if defs:
            pred = defs[0]
            if pred.is_predicate_reg:
                self._set_val(IRBuilder, IRRegs, pred, self.ir.Constant(self.ir.IntType(1), 1))


    @lift_for("ATOMG")
    def lift_atomg(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
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
        
        ptr = _as_pointer(
            addr_val,
            elem_ty,
            "atomg_ptr",
            addrspace=1,
        )
        
        cas_ord = "monotonic"
        cas_fail_ord = "monotonic"
        if op_type in ("CAS", "CAST"):
            cmp_val = _get_val(uses[1], "atomg_cmp")
            new_val = _get_val(uses[2], "atomg_new")

            cmp_val = self._insert_type_cast_if_needed(IRBuilder, cmp_val, elem_ty, "atomg_cmp")
            new_val = self._insert_type_cast_if_needed(IRBuilder, new_val, elem_ty, "atomg_new")

            # System-scope CAS (returns old value) -> nvvm .sys intrinsic, for
            # cross-device (host<->GPU) coherence on managed memory. Plain
            # cmpxchg lowers to device scope and loses host-concurrent updates.
            sys_cas = None
            if "SYS" in modifiers and op_type == "CAS":
                sys_cas = self._atom_sys_intrinsic(
                    IRBuilder, ptr, "CAS", new_val, cmp_val=cmp_val, name="atomg_sys_cas")
            if sys_cas is not None:
                old_val = sys_cas
            else:
                cas_result = IRBuilder.cmpxchg(ptr, cmp_val, new_val, cas_ord, cas_fail_ord, "atomg_cas")
                if op_type == "CAST":
                    # ATOMG.CAST returns success bit, not old value — see lift_atoms.
                    success_i1 = IRBuilder.extract_value(cas_result, 1, "atomg_cas_succ")
                    old_val = IRBuilder.zext(success_i1, elem_ty, "atomg_cas_succ_ext")
                else:
                    old_val = IRBuilder.extract_value(cas_result, 0, "atomg_cas_old")
        else:
            # CUDA atomicInc(addr,val) / atomicDec(addr,val) are wrap-around RMWs,
            # NOT a plain +1/-1: INC = (old >= val) ? 0 : old+1 (LLVM uinc_wrap);
            # DEC = ((old==0)||(old>val)) ? val : old-1 (LLVM udec_wrap). The wrap
            # limit `val` is an explicit register operand (ATOMG.E.INC ... R16,
            # [addr], R19 -> R19 is the limit). Lowering these as `add 1`/`sub 1`
            # breaks any grid-wide last-block barrier (bh's BoundingBoxKernel uses
            # `if (inc == atomicInc(blkcntd, inc))`): without the wrap the counter
            # is never reset to 0, so on the 2nd+ kernel launch no block is ever
            # "the last block" -> radius/root never computed -> TreeBuildingKernel
            # spins on an uninitialised tree -> hang at multi-step runs.
            rmw_ops = {
                "ADD": "add", "MIN": "min", "MAX": "max",
                "AND": "and", "OR": "or", "XOR": "xor",
                "EXCH": "xchg", "INC": "uinc_wrap", "DEC": "udec_wrap",
            }
            rmw_op = rmw_ops.get(op_type)
            if rmw_op is None:
                raise UnsupportedInstructionException(f"Unsupported ATOMG operation: {op_type}")

            src_val = _get_val(uses[1], "atomg_val")
            src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "atomg_val")

            # System-scope rmw -> nvvm .sys intrinsic (cross-device coherence).
            sys_rmw = None
            if "SYS" in modifiers and op_type not in ("INC", "DEC"):
                sys_rmw = self._atom_sys_intrinsic(
                    IRBuilder, ptr, op_type, src_val, name=f"atomg_sys_{op_type.lower()}")
            if sys_rmw is not None:
                old_val = sys_rmw
            else:
                old_val = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "monotonic", f"atomg_{op_type.lower()}")
        
        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            dest_ty = dest_reg.get_ir_type(self)
            old_val = self._insert_type_cast_if_needed(IRBuilder, old_val, dest_ty, "atomg_result")
            self._set_val(IRBuilder, IRRegs, dest_reg, old_val)

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
        
        shared_addr = self._shared_gep(IRBuilder, addr_val, "atoms_shared_addr")
        ptr = IRBuilder.bitcast(shared_addr, self.ir.PointerType(elem_ty, 3), "atoms_ptr")
        
        if op_type in ("CAS", "CAST"):
            cmp_val = _get_val(uses[1], "atoms_cmp")
            new_val = _get_val(uses[2], "atoms_new")

            cmp_val = self._insert_type_cast_if_needed(IRBuilder, cmp_val, elem_ty, "atoms_cmp")
            new_val = self._insert_type_cast_if_needed(IRBuilder, new_val, elem_ty, "atoms_new")

            cas_result = IRBuilder.cmpxchg(ptr, cmp_val, new_val, "monotonic", "monotonic", "atoms_cas")
            if op_type == "CAST":
                # SASS ATOMS.CAST returns the success bit (1/0), not the old
                # value — ptxas's spin-CAS idiom tests `R == 1` to decide
                # whether to retry. Mirrors the CPU lifter fix.
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

            # ATOMS.POPC.INC/DEC encodes the operand list as just the address
            # (the increment is implicit: each participating lane contributes
            # 1, hardware aggregates popcount across the warp). Other RMW
            # variants carry the source value as uses[1]. Mirror lifter_cpu's
            # ordering so an INC with no uses[1] doesn't IndexError.
            if op_type in ("INC", "DEC"):
                src_val = self.ir.Constant(elem_ty, 1)
            else:
                src_val = _get_val(uses[1], "atoms_val")
                src_val = self._insert_type_cast_if_needed(IRBuilder, src_val, elem_ty, "atoms_val")

            old_val = IRBuilder.atomic_rmw(rmw_op, ptr, src_val, "monotonic", f"atoms_{op_type.lower()}")
        
        if dest_reg and dest_reg.is_writable_reg and not dest_reg.is_rz:
            # A 64-bit ATOMS dest (e.g. ATOMS.CAST.SPIN.64) is a reg_group
            # {R_lo:R_hi}; let _set_val split the full 64-bit result across the
            # pair (zeroing R_hi for the CAST success result). Casting to the
            # single-reg get_ir_type (i32) here would drop the high word and
            # leave R_hi stale — the spin-loop .EX (R_hi == 0) test then never
            # passes -> deadlock.
            if not dest_reg.is_reg_group:
                dest_ty = dest_reg.get_ir_type(self)
                old_val = self._insert_type_cast_if_needed(IRBuilder, old_val, dest_ty, "atoms_result")
            self._set_val(IRBuilder, IRRegs, dest_reg, old_val)

    @lift_for("SYNCS")
    def lift_syncs(self, Inst, IRBuilder, IRRegs, block_map, _get_val=None, _as_pointer=None):
        """SM90 SYNCS mbarrier ops -> real PTX mbarrier instructions.

        The Hopper mbarrier is a 64-bit object in shared memory; ptxas compiles
        the PTX mbarrier.* ops to exactly this SYNCS family, so lifting back to
        the PTX ops round-trips the semantics:

          SYNCS.EXCH.64 URZ,[a],Rv   -> mbarrier.init with the encoded count,
              followed by a full-count arrive if the initial phase is one.
          SYNCS.ARRIVE.TRANS64[.RED].A1T0 RZ,[a],RZ -> mbarrier.arrive
          SYNCS.ARRIVE.TRANS64[.RED].A0T1 RZ,[a],Rt -> mbarrier.arrive.expect_tx
              (arrive-count 0, bump expected-transaction count: the TMA
              complete_tx path decrements it as bulk copies land.)
          SYNCS.PHASECHK.TRANS64[.TRYWAIT] P,[a],Rp -> mbarrier.try_wait.parity /
              mbarrier.test_wait.parity (predicate dest = phase complete; the
              SASS spin loop around it is lifted as-is).
          CCTL / RLSE / other cache-control forms: NOP.

        The address operand is a shared-memory byte address exactly like
        LDS/ATOMS — reuse _shared_gep so the dynamic-shared base and cluster
        window handling stay identical.
        """
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        if not uses:
            return
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)

        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            return
        addr_val = _get_val(mem_ops[0], "mbar_addr")
        gep = self._shared_gep(IRBuilder, addr_val, "mbar_shared")
        bar_ptr = IRBuilder.ptrtoint(gep, i64, "mbar_ptr")
        other = [op for op in uses if not op.is_mem_addr]

        def _val_or_zero(ty, nm):
            if other:
                v = _get_val(other[0], nm)
                return self._insert_type_cast_if_needed(IRBuilder, v, ty, nm)
            return self.ir.Constant(ty, 0)

        if "EXCH" in modifiers:
            # mbarrier.init compiles to SYNCS.EXCH.64 of the ENCODED state.
            # Observed encoding (ptxas SASS for init count=N, ntid=256 ->
            # lo=0x1FFE00): lo word = (2^20 - N) << 1 — the arrival count is
            # stored as a 20-bit COUNTDOWN biased at 2^20, pre-shifted by 1.
            # A raw atom.exch would go through the MEMORY unit (ATOMS) and the
            # BARRIER unit never sees a valid init -> "Illegal barrier arrive".
            # Decode N = 2^20 - (lo >> 1) from the lifted dataflow and emit a
            # real mbarrier.init so the sync unit installs the state itself.
            state = _val_or_zero(i64, "mbar_init_state")
            low = IRBuilder.trunc(state, i32, "mbar_init_low")
            half = IRBuilder.lshr(low, self.ir.Constant(i32, 1), "mbar_cd")
            count = IRBuilder.sub(
                self.ir.Constant(i32, 1 << 20), half, "mbar_count_raw")
            count = IRBuilder.and_(
                count, self.ir.Constant(i32, (1 << 20) - 1), "mbar_count")
            phase = IRBuilder.trunc(
                IRBuilder.lshr(state, self.ir.Constant(i64, 63)), i32, "mbar_init_phase")
            # PTX init starts in phase zero. A counted arrival completes that
            # phase and restores the pending count for phase one. Uniform SASS
            # performs the exchange once for the active warp.
            if defs and defs[0].is_uniform_reg:
                init = (".reg .b32 mask, enabled; .reg .pred leader;"
                        " activemask.b32 mask; elect.sync _|leader, mask;"
                        " @leader mbarrier.init.shared::cta.b64 [$0], $1;"
                        " selp.u32 enabled, $2, 0, leader; setp.ne.u32 advance, enabled, 0;")
            else:
                init = "mbarrier.init.shared::cta.b64 [$0], $1; setp.ne.u32 advance, $2, 0;"
            asm_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i32, i32])
            asm = self.ir.InlineAsm(
                asm_ty, "{ .reg .pred advance; " + init +
                " @advance mbarrier.arrive.shared::cta.b64 _, [$0], $1; }",
                "l,r,r,~{memory}",
                side_effect=True)
            IRBuilder.call(asm, [bar_ptr, count, phase], "")
            return

        if "ARRIVE" in modifiers:
            # ARRIVES may target a PEER CTA's barrier: SASS addresses it via
            # the DSMEM cluster-window encoding (rank<<24)|offset. _shared_gep
            # masked the rank bits to form the LOCAL pointer; recover the rank
            # from the RAW address and, for a GENUINE peer arrive, map the
            # local shared::cta address into the target CTA's window with mapa.
            # SELF-arrives (rank byte == %cluster_ctarank, which includes every
            # arrive in a NON-cluster launch) MUST use the plain ::cta form on
            # the local pointer: mapa/::cluster addressing is undefined outside
            # a real cluster launch — runtime-verified on the paged FA3 fwd
            # kernel (cluster C<1,1,1>, plain launch), where the mapa'd arrives
            # all landed on one bogus word (0x28c58) and every real barrier
            # stayed pristine -> full producer/consumer deadlock. Conversely
            # FA3 C<2,1,1> needs the mapa path for its rank!=self arrives, so
            # select per-arrive at runtime on (rank == own rank).
            addr64 = self._insert_type_cast_if_needed(
                IRBuilder, addr_val, i64, "mbar_addr64")
            rank = IRBuilder.and_(
                IRBuilder.lshr(addr64, self.ir.Constant(i64, 24), "mbar_rank_sh"),
                self.ir.Constant(i64, 0xFF), "mbar_rank64")
            rank32 = IRBuilder.trunc(rank, i32, "mbar_rank")
            bar32 = IRBuilder.trunc(bar_ptr, i32, "mbar_ptr32")

            # SASS suffix decode (each mapping VERIFIED BY PTX MICROTEST,
            # scratchpad mbar_micro*.ptx 2026-07-12 — never guess these):
            #   A1T0 RZ,[a],RZ  <- mbarrier.arrive...b64 s,[a]        (arrive 1)
            #   A0T1 RZ,[a],RZ  <- NOT a source-level op: ptxas lowers ONE
            #       cp.async.mbarrier.arrive into the PAIR
            #       {SYNCS.ARRIVE.TRANS64.RED.A0T1 ; ARRIVES.LDGSTSBAR.64
            #       .TRANSCNT} (microtest kcparr). The ARRIVES.LDGSTSBAR lift
            #       already re-emits cp.async.mbarrier.arrive, which
            #       regenerates BOTH — so the A0T1 lifts to NOTHING. Lifting
            #       it as a separate expect_tx double-books the expected
            #       transaction units and the phase never completes (paged
            #       FA3 fwd deadlock, 2026-07-12).
            #   no-suffix RZ,[a],Rt (live Rt) <- mbarrier.arrive.expect_tx
            #       ...b64 s,[a],t (TMA bulk: arrive 1 + expect tx bytes;
            #       tx==0 at runtime is ILLEGAL, but live-src sites always
            #       carry real byte counts).
            src_is_live = other and not getattr(other[0], "is_rz", False) and \
                getattr(other[0], "reg", None) not in ("RZ", "URZ")
            has_live_dest = any(d.is_writable_reg and not d.is_rz for d in defs)
            if "A0T1" in modifiers and not src_is_live:
                return
            if src_is_live and "A1T0" not in modifiers:
                tx = _val_or_zero(i32, "mbar_tx")
                asm_ty = self.ir.FunctionType(
                    self.ir.VoidType(), [i32, i64, i32, i32])
                asm = self.ir.InlineAsm(
                    asm_ty,
                    "{.reg .pred p; .reg .u32 r; .reg .u32 m; .reg .b64 s; "
                    "mov.u32 r, %cluster_ctarank; setp.ne.u32 p, r, $2; "
                    "@p mapa.shared::cluster.u32 m, $0, $2; "
                    "@p mbarrier.arrive.expect_tx.release.cluster.shared::cluster.b64 _, [m], $3; "
                    "@!p mbarrier.arrive.expect_tx.shared::cta.b64 s, [$1], $3;}",
                    "r,l,r,r,~{memory}", side_effect=True)
                IRBuilder.call(asm, [bar32, bar_ptr, rank32, tx], "")
            elif not has_live_dest:
                # Sink-dest arrive (SASS RZ dest): local ::cta form for
                # self-arrives, mapa + ::cluster form for peer arrives.
                asm_ty = self.ir.FunctionType(self.ir.VoidType(), [i32, i64, i32])
                asm = self.ir.InlineAsm(
                    asm_ty,
                    "{.reg .pred p; .reg .u32 r; .reg .u32 m; .reg .b64 s; "
                    "mov.u32 r, %cluster_ctarank; setp.ne.u32 p, r, $2; "
                    "@p mapa.shared::cluster.u32 m, $0, $2; "
                    "@p mbarrier.arrive.release.cluster.shared::cluster.b64 _, [m]; "
                    "@!p mbarrier.arrive.shared::cta.b64 s, [$1];}",
                    "r,l,r,~{memory}", side_effect=True)
                IRBuilder.call(asm, [bar32, bar_ptr, rank32], "")
            else:
                # State-returning arrive exists only in the ::cta form; such
                # arrives are always on the CTA's own barrier (the state is
                # meaningless cross-CTA), so the local pointer is correct.
                asm_ty = self.ir.FunctionType(i64, [i64])
                asm = self.ir.InlineAsm(
                    asm_ty, "mbarrier.arrive.shared::cta.b64 $0, [$1];", "=l,l,~{memory}",
                    side_effect=True)
                state = IRBuilder.call(asm, [bar_ptr], "mbar_arrive")
                for d in defs:
                    if d.is_writable_reg and not d.is_rz:
                        dest_ty = d.get_ir_type(self)
                        v = self._insert_type_cast_if_needed(
                            IRBuilder, state, dest_ty, "mbar_state")
                        self._set_val(IRBuilder, IRRegs, d, v)
                        break
            return

        if "PHASECHK" in modifiers:
            # SASS carries the phase parity in BIT 31 of the operand register
            # (ptxas shifts the PTX-level {0,1} parity up with SHF.L 31 before
            # SYNCS.PHASECHK); PTX mbarrier.*_wait.parity reads bit 0. Decode
            # with a logical >>31 or phase 1 waits on the wrong parity forever.
            par = _val_or_zero(i32, "mbar_parity")
            par = IRBuilder.lshr(par, self.ir.Constant(i32, 31), "mbar_par_bit")
            op = "try_wait" if "TRYWAIT" in modifiers else "test_wait"
            asm_ty = self.ir.FunctionType(i32, [i64, i32])
            asm = self.ir.InlineAsm(
                asm_ty,
                "{.reg .pred p; mbarrier.%s.parity.shared::cta.b64 p, [$1], $2; "
                "selp.u32 $0, 1, 0, p;}" % op,
                "=r,l,r,~{memory}", side_effect=True)
            done = IRBuilder.call(asm, [bar_ptr, par], "mbar_phasechk")
            pred = IRBuilder.icmp_unsigned(
                "!=", done, self.ir.Constant(i32, 0), "mbar_ready")
            for d in defs:
                if d.is_predicate_reg and not d.is_pt:
                    self._set_val(IRBuilder, IRRegs, d, pred)
                    break
            return
        # CCTL.IV / RLSE / other cache-control hints: NOP.

    @lift_for("HGMMA", "WGMMA", "QGMMA")
    def lift_wgmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """SM90 HGMMA/WGMMA (SS descriptor form) -> PTX wgmma.mma_async.

        HGMMA.64xNx16.F32[.BF16] Rd, gdesc[URn], Rc [, !UPT, gsbK]
        The parser (sir/instruction.py) expands Rd/Rc into N/2-register f32
        accumulator GROUPS and synthesizes the descriptor words:
          URn (desc_a lo, tagged desc_reg), UR n+1 (desc_a hi, desc_a_hi),
          UR n+2 (desc_b lo, desc_hi),      UR n+3 (desc_b hi, desc_b_hi).
        The DESCRIPTOR VALUES are computed by the original kernel's own lifted
        dataflow (base>>4 | LBO<<16 | SBO<<32 | swizzle<<62) — pass the two
        reassembled 64-bit values straight into the asm; no re-encoding here.

        The C form selects scale-d: `Rc == RZ, !UPT` -> scale-d 0 (D = A*B);
        otherwise C is the same register group as D and is tied read-write into
        the accumulator operands (D = A*B + D).
        """
        opc = Inst.opcodes
        shape = opc[1] if len(opc) > 1 else "64x8x16"
        parts = shape.split("x")
        n_dim = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 8
        k_dim = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 16
        accum = opc[2] if len(opc) > 2 else "F32"
        # Input type from the trailing opcode tokens: HGMMA carries F32[.BF16]
        # (F16 implied), QGMMA (fp8) spells both operand types, e.g.
        # QGMMA.64x128x32.F32.E4M3.E4M3.
        # Input types from trailing opcode tokens. HGMMA: F32[.BF16|.TF32]
        # (F16 implied). QGMMA spells BOTH operand types and they can DIFFER
        # (57_hopper_grouped: QGMMA.64x128x32.F32.E4M3.E5M2 — A=e4m3,
        # B=e5m2); collect in order.
        tys = []
        for tok in opc[3:]:
            t = tok.upper()
            if t.startswith("BF16"):
                tys.append("bf16")
            elif t.startswith("E4M3"):
                tys.append("e4m3")
            elif t.startswith("E5M2"):
                tys.append("e5m2")
            elif t.startswith("TF32"):
                tys.append("tf32")
        in_ty_a = tys[0] if tys else "f16"
        in_ty_b = tys[1] if len(tys) > 1 else in_ty_a
        in_ty = in_ty_a  # kept for messages
        # fp8 AND tf32 wgmma take no imm-trans operands (K-major only).
        is_fp8 = in_ty_a in ("e4m3", "e5m2", "tf32")
        if not accum.startswith("F32"):
            raise UnsupportedInstructionException(
                f"WGMMA accumulator {accum} not supported on nvvm yet: {opc}")
        d_count = max(n_dim // 2, 4)

        defs = Inst.get_defs()
        uses = Inst.get_uses()
        if len(defs) < d_count:
            raise UnsupportedInstructionException(
                f"WGMMA def group short: {len(defs)} < {d_count}: {opc}")
        if all(getattr(d, "reg", None) == "RZ" for d in defs):
            return  # RZ dest: barrier form, no-op

        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        f32 = self.ir.FloatType()

        # Descriptor words by parser tag. RS form (A from registers): the
        # parser tags the 4-reg A fragment `wgmma_a_frag` and resolves the
        # B-descriptor UR pair by convention — {URn+2, URn+3} for .tnspB RS
        # (same window as SS), {URn, URn+1} for plain RS (the encoded pair
        # IS the descriptor). See sir/instruction.py for the evidence.
        d_lo = d_ahi = d_blo = d_bhi = None
        a_frag = []
        c_regs = []
        for u in uses:
            if getattr(u, "wgmma_a_frag", False):
                a_frag.append(u)
            elif getattr(u, "desc_b_hi", False):
                d_bhi = u
            elif getattr(u, "desc_a_hi", False):
                d_ahi = u
            elif getattr(u, "desc_hi", False):
                d_blo = u
            elif getattr(u, "desc_reg", None) or (u.name and "gdesc" in str(u.name)):
                d_lo = u
            elif u.is_reg and not u.is_predicate_reg:
                c_regs.append(u)
        is_rs = bool(a_frag)
        if d_lo is None or d_blo is None:
            raise UnsupportedInstructionException(
                f"WGMMA descriptor operands missing: {opc}")

        def _desc64(lo_op, hi_op, nm):
            lo = _get_val(lo_op, nm + "_lo")
            lo = self._insert_type_cast_if_needed(IRBuilder, lo, i32, nm + "_lo32")
            lo64 = IRBuilder.zext(lo, i64, nm + "_lo64")
            if hi_op is None:
                return lo64
            hi = _get_val(hi_op, nm + "_hi")
            hi = self._insert_type_cast_if_needed(IRBuilder, hi, i32, nm + "_hi32")
            hi64 = IRBuilder.shl(IRBuilder.zext(hi, i64, nm + "_hi64"),
                                 self.ir.Constant(i64, 32), nm + "_hishift")
            return IRBuilder.or_(lo64, hi64, nm)

        if is_rs:
            # RS: only B has a descriptor; the parser already tagged the
            # right UR pair for the tnspB/plain convention (runtime-proven
            # both ways: fwd fp16 P@V .tnspB needed +2 — identity-V
            # instrumentation showed {URn,URn+1} held the stale Q-desc — and
            # bwd hdim96 plain dP needed +0, its desc ULDC'd directly at URn).
            desc_a = None
            desc_b = _desc64(d_blo, d_bhi, "wgmma_desc_b")
        else:
            desc_a = _desc64(d_lo, d_ahi, "wgmma_desc_a")
            desc_b = _desc64(d_blo, d_bhi, "wgmma_desc_b")

        # REBASE the start-address field. The original descriptor's 14-bit base
        # (bits 0..13) is a window-absolute smem address >>4 from the ORIGINAL
        # module's layout. Our lifted stores go through _shared_gep(@shared_mem,
        # v) = ptrtoint(@shared_mem) + v - _shared_dyn_base, so the data lives
        # at the original window address SHIFTED by (smbase - dyn_base). Apply
        # the SAME delta to the descriptor base so wgmma reads where the lifted
        # code actually wrote. Static kernels: dyn_base = 0, delta = smbase
        # (validated wgmma microbench behavior, unchanged). Dynamic-shared
        # kernels (FA3): @shared_mem is the extern[] dynamic region the JIT
        # places right after the reserved window region — exactly where the
        # original's dynamic data began — so the delta is 0 and the original
        # descriptor value is already correct. Offsets stay within the 14-bit
        # field for any realistic smem size (<=228KB -> <=14250).
        sm_base = IRBuilder.ptrtoint(self.SharedMem, i64, "wgmma_smbase")
        dyn = getattr(self, "_shared_dyn_base", 0)
        if dyn:
            sm_base = IRBuilder.sub(
                sm_base, self.ir.Constant(i64, dyn), "wgmma_smbase_dyn")
        sm_base16 = IRBuilder.lshr(sm_base, self.ir.Constant(i64, 4), "wgmma_smbase16")
        # Clamp the delta to the descriptor's 14-bit start-address field.
        # Under a CLUSTER launch each CTA's @shared_mem VA carries its DSMEM
        # rank bits ((rank<<24)|off): unmasked, rank 1's delta = 0x100000 and
        # the add corrupted the LBO field (bit 20) -> every rank-1 CTA read V
        # with a wrong leading-dim offset (SQ=256 C<2,1,1>: rows 128-255 all
        # wrong, rows 0-127 bit-exact). The rank bits are not part of the
        # descriptor encoding — the base field is window-relative.
        sm_base16 = IRBuilder.and_(
            sm_base16, self.ir.Constant(i64, 0x3FFF), "wgmma_smbase16_fld")
        if desc_a is not None:
            desc_a = IRBuilder.add(desc_a, sm_base16, "wgmma_desc_a_reb")
        desc_b = IRBuilder.add(desc_b, sm_base16, "wgmma_desc_b_reb")

        # scale-d: 0 when C is the RZ,!UPT zero-init form.
        c_is_rz = (not c_regs) or all(getattr(c, "reg", None) == "RZ"
                                      for c in c_regs[:d_count])
        # trans a/b from gdesc modifiers (K-major default = 0).
        dname = str(getattr(d_lo, "name", "") or "")
        t_a = 1 if "tnspA" in dname else 0
        t_b = 1 if "tnspB" in dname else 0

        regs = "{" + ",".join(f"${i}" for i in range(d_count)) + "}"
        ret_ty = self.ir.LiteralStructType([f32] * d_count)
        if is_rs:
            # RS form: A = 4 packed-f16x2 .b32 regs from the fragment; only B
            # has a descriptor and only imm-trans-b applies.
            a_vals = []
            for a in a_frag[:4]:
                v = _get_val(a, "wgmma_a")
                v = self._insert_type_cast_if_needed(IRBuilder, v, i32, "wgmma_a32")
                a_vals.append(v)
            while len(a_vals) < 4:
                a_vals.append(self.ir.Constant(i32, 0))
            abrace = "{" + ",".join(f"${d_count + i}" for i in range(4)) + "}"
            if c_is_rz:
                trs = "" if is_fp8 else f", {t_b}"
                asm_str = (f"wgmma.mma_async.sync.aligned.m64n{n_dim}k{k_dim}.f32.{in_ty_a}.{in_ty_b} "
                           f"{regs}, {abrace}, ${d_count + 4}, 0, 1, 1{trs};")
                cons = ",".join(["=f"] * d_count) + ",r,r,r,r,l,~{memory}"
                fn_ty = self.ir.FunctionType(ret_ty, [i32] * 4 + [i64])
                asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
                res = IRBuilder.call(asm, a_vals + [desc_b], "wgmma_rs")
            else:
                trs = "" if is_fp8 else f", {t_b}"
                asm_str = (f"wgmma.mma_async.sync.aligned.m64n{n_dim}k{k_dim}.f32.{in_ty_a}.{in_ty_b} "
                           f"{regs}, {abrace}, ${d_count + 4}, 1, 1, 1{trs};")
                cons = ",".join(["=f"] * d_count) + ",r,r,r,r,l," + \
                       ",".join(str(i) for i in range(d_count))
                c_vals = []
                for c in c_regs[:d_count]:
                    v = _get_val(c, "wgmma_c")
                    v = self._insert_type_cast_if_needed(IRBuilder, v, f32, "wgmma_c32")
                    c_vals.append(v)
                while len(c_vals) < d_count:
                    c_vals.append(self.ir.Constant(f32, 0))
                fn_ty = self.ir.FunctionType(ret_ty, [i32] * 4 + [i64] + [f32] * d_count)
                asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
                res = IRBuilder.call(asm, a_vals + [desc_b] + c_vals, "wgmma_rs")
        elif c_is_rz:
            trs = "" if is_fp8 else f", {t_a}, {t_b}"
            asm_str = (f"wgmma.mma_async.sync.aligned.m64n{n_dim}k{k_dim}.f32.{in_ty_a}.{in_ty_b} "
                       f"{regs}, ${d_count}, ${d_count + 1}, 0, 1, 1{trs};")
            cons = ",".join(["=f"] * d_count) + ",l,l,~{memory}"
            fn_ty = self.ir.FunctionType(ret_ty, [i64, i64])
            asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
            res = IRBuilder.call(asm, [desc_a, desc_b], "wgmma")
        else:
            # Accumulate: tie each C value read-write into its D slot.
            trs = "" if is_fp8 else f", {t_a}, {t_b}"
            asm_str = (f"wgmma.mma_async.sync.aligned.m64n{n_dim}k{k_dim}.f32.{in_ty_a}.{in_ty_b} "
                       f"{regs}, ${d_count}, ${d_count + 1}, 1, 1, 1{trs};")
            cons = ",".join(["=f"] * d_count) + ",l,l," + \
                   ",".join(str(i) for i in range(d_count))
            c_vals = []
            for c in c_regs[:d_count]:
                v = _get_val(c, "wgmma_c")
                v = self._insert_type_cast_if_needed(IRBuilder, v, f32, "wgmma_c32")
                c_vals.append(v)
            while len(c_vals) < d_count:
                c_vals.append(self.ir.Constant(f32, 0))
            fn_ty = self.ir.FunctionType(ret_ty, [i64, i64] + [f32] * d_count)
            asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
            res = IRBuilder.call(asm, [desc_a, desc_b] + c_vals, "wgmma")

        for i, d in enumerate(defs[:d_count]):
            if getattr(d, "reg", None) == "RZ":
                continue
            v = IRBuilder.extract_value(res, i, f"wgmma_d{i}")
            self._set_val(IRBuilder, IRRegs, d, v)

    @lift_for("UTMALDG")
    def lift_utmaldg(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UTMALDG.<N>D -> cp.async.bulk.tensor.Nd (TMA tile load).

        SASS (coords-inline form): UTMALDG.ND [URt], [URd] where URt is a UR
        TUPLE {smem dest, mbarrier, coord0..coordN-1} (parser synthesizes the
        members with tma_tuple_idx tags) and URd:URd+1 = the 64-bit generic
        address of the CUtensorMap. smem/mbar are rebased through _shared_gep
        like every other shared access; the tensormap pointer value passes
        through from the lifted dataflow (the driver must make the same
        tensormap available at that address — FA-C's job).
        """
        mods = [t.upper() for t in Inst.opcodes[1:]]
        nd = next((int(t[:-1]) for t in mods if t.endswith("D") and t[:-1].isdigit()), 2)
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)

        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            raise UnsupportedInstructionException(f"UTMALDG without mem ops: {Inst.opcodes}")
        smem_v = _get_val(mem_ops[0], "tma_smem")
        smem_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, smem_v, "tma_smem_gep"), i64, "tma_smem_ptr")

        # Tensormap pointer: the first mem operand after the tuple that is NOT
        # the desc[URm] cache/access memdesc (tagged tma_memdesc at parse).
        # Mistaking the memdesc for the tensormap (its value is the constant
        # {0, 0x14f00000} default policy) made TMA read the tensormap from
        # garbage -> warp out-of-range fault on every K/V tile load.
        desc_op = next(
            (op for op in mem_ops[1:] if not getattr(op, "tma_memdesc", False)),
            None)
        if desc_op is None:
            raise UnsupportedInstructionException(f"UTMALDG tensormap missing: {Inst.opcodes}")
        desc_v = _get_val(desc_op, "tma_desc")
        desc_v = self._insert_type_cast_if_needed(IRBuilder, desc_v, i64, "tma_desc64")

        mbar_v = None
        coords = []
        for u in uses:
            j = getattr(u, "tma_tuple_idx", None)
            if j == 1:
                mbar_v = _get_val(u, "tma_mbar")
            elif j is not None and j >= 2:
                c = _get_val(u, f"tma_c{j-2}")
                coords.append(self._insert_type_cast_if_needed(IRBuilder, c, i32, f"tma_c{j-2}_32"))
        if mbar_v is None:
            raise UnsupportedInstructionException(f"UTMALDG mbarrier member missing: {Inst.opcodes}")
        mbar_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, mbar_v, "tma_mbar_gep"), i64, "tma_mbar_ptr")
        while len(coords) < nd:
            coords.append(self.ir.Constant(i32, 0))
        coords = coords[:nd]

        # MULTICAST: the CTA-mask is the plain (non-mem, non-tuple, non-pred)
        # reg use. PTX takes it as a 16-bit operand on the .multicast::cluster
        # form; the TMA unit then delivers each CTA's slice to every CTA in the
        # mask and fires complete_tx on each receiver's own mbarrier.
        mask_v = None
        if any(t.upper() == "MULTICAST" for t in Inst.opcodes[1:]):
            mask_op = next(
                (u for u in uses
                 if u.is_reg and not u.is_mem_addr
                 and not getattr(u, "is_predicate_reg", False)
                 and getattr(u, "tma_tuple_idx", None) is None
                 and getattr(u, "reg", None) not in (None, "RZ", "URZ")),
                None)
            if mask_op is not None:
                mv = _get_val(mask_op, "tma_ctamask")
                mv = self._insert_type_cast_if_needed(IRBuilder, mv, i32, "tma_ctamask32")
                # The SASS mask operand carries the CTA mask in bits 16-31
                # (`UMOV UR19, 0x30000` = mask 0b11 for a 2-CTA cluster);
                # truncating the raw value zeroed the PTX .b16 ctaMask -> the
                # multicast delivered to NO CTA and every consumer deadlocked
                # on complete_tx (runtime-dumped mask slots were all 0).
                mv = IRBuilder.lshr(mv, self.ir.Constant(i32, 16), "tma_ctamask_sh")
                mask_v = IRBuilder.trunc(mv, self.ir.IntType(16), "tma_ctamask16")

        cbrace = "{" + ",".join(f"${3 + i}" for i in range(nd)) + "}"
        if mask_v is not None:
            body = (f"cp.async.bulk.tensor.{nd}d.shared::cluster.global"
                    f".mbarrier::complete_tx::bytes.multicast::cluster"
                    f" [$0], [$1, {cbrace}], [$2], ${3 + nd}")
            cons = "l,l,l," + ",".join(["r"] * nd) + ",h,~{memory}"
            fn_ty = self.ir.FunctionType(
                self.ir.VoidType(),
                [i64, i64, i64] + [i32] * nd + [self.ir.IntType(16)])
            asm = self.ir.InlineAsm(
                fn_ty, self._elect_one_wrap(body), cons, side_effect=True)
            IRBuilder.call(asm, [smem_ptr, desc_v, mbar_ptr] + coords + [mask_v], "")
        else:
            body = (f"cp.async.bulk.tensor.{nd}d.shared::cluster.global"
                    f".mbarrier::complete_tx::bytes [$0], [$1, {cbrace}], [$2]")
            cons = "l,l,l," + ",".join(["r"] * nd) + ",~{memory}"
            fn_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i64, i64] + [i32] * nd)
            asm = self.ir.InlineAsm(
                fn_ty, self._elect_one_wrap(body), cons, side_effect=True)
            IRBuilder.call(asm, [smem_ptr, desc_v, mbar_ptr] + coords, "")

    @staticmethod
    def _elect_one_wrap(op_str):
        """Wrap a PTX op so it executes ONCE PER WARP (one elected lane).

        SASS U-prefixed TMA ops (UTMALDG/UTMASTG) run on the UNIFORM datapath:
        one execution per warp that reaches them, regardless of active lanes.
        The PTX cp.async.bulk.tensor equivalent is a regular per-THREAD
        instruction — unguarded, every active lane of the producer warp
        issues the copy, over-delivering complete_tx bytes on the mbarrier
        (32x the expected transaction count corrupts the phase and the whole
        producer/consumer pipeline deadlocks in try_wait — observed on FA3
        fwd). elect.sync over %activemask picks exactly one lane, matching
        the uniform-datapath semantics (same recipe as lift_elect)."""
        return ("{ .reg .pred pe; .reg .b32 me; activemask.b32 me; "
                f"elect.sync _|pe, me; @pe {op_str}; }}")

    @lift_for("LDGSTS")
    def lift_ldgsts(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """LDGSTS (Ampere+ async global->shared copy) -> PTX cp.async.

        LDGSTS.E.BYPASS.LTC128B.128 [Rsm+off], desc[URd][Rg.64+off], !Pz
        The .128/.64/.32 modifier is the copy size (16/8/4 bytes). When it is
        omitted, the copy is 4 bytes, including with the .LTC128B cache hint.
        16B copies use .cg (L1-bypass, matching .BYPASS), smaller .ca. The trailing
        predicate is the SOURCE-read guard: when it evaluates FALSE the copy
        ZERO-FILLS the destination instead of reading global — lower it as the
        cp.async src-size operand (pred ? size : 0). Completion is tracked by
        the kernel's own ARRIVES.LDGSTSBAR (-> cp.async.mbarrier.arrive), so no
        commit/wait here.
        """
        uses = Inst.get_uses()
        mem_ops = [op for op in uses if op.is_mem_addr]
        if len(mem_ops) < 2:
            raise UnsupportedInstructionException(f"LDGSTS needs 2 mem ops: {Inst.opcodes}")
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        size = 4
        for m in Inst.opcodes[1:]:
            if m in ("128", "64", "32"):
                size = int(m) // 8
                break
        smem_v = _get_val(mem_ops[0], "cpa_smem")
        smem_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, smem_v, "cpa_smem_gep"), i64, "cpa_smem_ptr")
        gbl = _get_val(mem_ops[1], "cpa_gbl")
        gbl = self._insert_type_cast_if_needed(IRBuilder, gbl, i64, "cpa_gbl64")

        # Source-read guard: a trailing predicate USE (e.g. `!P2`). The lifted
        # value is the already-negated boolean; src-size = pred ? size : 0.
        pred_ops = [op for op in uses if getattr(op, "is_predicate_reg", False)
                    and not getattr(op, "is_pt", False)]
        variant = "cg" if size == 16 else "ca"
        if pred_ops:
            pv = _get_val(pred_ops[-1], "cpa_zfill")
            src_sz = IRBuilder.select(
                pv, self.ir.Constant(i32, size), self.ir.Constant(i32, 0), "cpa_srcsz")
            asm = self.ir.InlineAsm(
                self.ir.FunctionType(self.ir.VoidType(), [i64, i64, i32]),
                f"cp.async.{variant}.shared.global [$0], [$1], {size}, $2;",
                "l,l,r,~{memory}", side_effect=True)
            IRBuilder.call(asm, [smem_ptr, gbl, src_sz], "")
        else:
            asm = self.ir.InlineAsm(
                self.ir.FunctionType(self.ir.VoidType(), [i64, i64]),
                f"cp.async.{variant}.shared.global [$0], [$1], {size};",
                "l,l,~{memory}", side_effect=True)
            IRBuilder.call(asm, [smem_ptr, gbl], "")

    @lift_for("ARRIVES")
    def lift_arrives(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """ARRIVES.LDGSTSBAR.64[.TRANSCNT] [mbar] ->
        cp.async.mbarrier.arrive[.noinc].shared::cta.b64: arrive on the
        mbarrier once all of this thread's prior cp.asyncs complete.

        ptxas mapping (verified by compiling both PTX forms, sm_90a/12.6):
          cp.async.mbarrier.arrive       -> ARRIVES.LDGSTSBAR.64.TRANSCNT
          cp.async.mbarrier.arrive.noinc -> ARRIVES.LDGSTSBAR.64.ARVCNT
        i.e. TRANSCNT is the PLAIN arrive (FA3/cutlass cpasync_barrier_arrive)
        and ARVCNT is the noinc form. (An earlier session briefly mapped
        TRANSCNT->noinc — backwards: it masked the FA3-paged deadlock but made
        the barrier phase complete before the cp.asyncs landed, showing up as
        intermittent stale-Q reads in packgqa.)"""
        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        uses = Inst.get_uses()
        mem_ops = [op for op in uses if op.is_mem_addr]
        if not mem_ops:
            return
        i64 = self.ir.IntType(64)
        v = _get_val(mem_ops[0], "cparr_addr")
        mbar = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, v, "cparr_gep"), i64, "cparr_ptr")
        noinc = ".noinc" if "ARVCNT" in modifiers else ""
        asm = self.ir.InlineAsm(
            self.ir.FunctionType(self.ir.VoidType(), [i64]),
            f"cp.async.mbarrier.arrive{noinc}.shared::cta.b64 [$0];",
            "l,~{memory}", side_effect=True)
        IRBuilder.call(asm, [mbar], "")

    @lift_for("UTMASTG")
    def lift_utmastg(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UTMASTG.<N>D [URt],[URd] -> cp.async.bulk.tensor.Nd.global.shared::cta
        .bulk_group (TMA tile STORE, FA epilogue O write). URt tuple =
        {smem src, coord0..coordN-1} (no mbarrier — completion is tracked via
        bulk_group: emit commit_group right after; the SASS ACQBULK maps to
        wait_group 0 already)."""
        mods = [t.upper() for t in Inst.opcodes[1:]]
        nd = next((int(t[:-1]) for t in mods if t.endswith("D") and t[:-1].isdigit()), 2)
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        mem_ops = [op for op in uses if op.is_mem_addr]
        if len(mem_ops) < 2:
            raise UnsupportedInstructionException(f"UTMASTG needs 2 mem ops: {Inst.opcodes}")
        smem_v = _get_val(mem_ops[0], "tmas_smem")
        smem_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, smem_v, "tmas_smem_gep"), i64, "tmas_smem_ptr")
        # Tensormap = first non-memdesc mem operand after the tuple (see
        # lift_utmaldg: desc[URm] is the cache/access policy, not the map).
        desc_op = next(
            (op for op in mem_ops[1:] if not getattr(op, "tma_memdesc", False)),
            None)
        if desc_op is None:
            raise UnsupportedInstructionException(f"UTMASTG tensormap missing: {Inst.opcodes}")
        desc_v = _get_val(desc_op, "tmas_desc")
        desc_v = self._insert_type_cast_if_needed(IRBuilder, desc_v, i64, "tmas_desc64")
        coords = []
        for u in uses:
            j = getattr(u, "tma_tuple_idx", None)
            if j is not None and j >= 2:
                c = _get_val(u, f"tmas_c{j-2}")
                coords.append(self._insert_type_cast_if_needed(IRBuilder, c, i32, f"tmas_c{j-2}_32"))
        while len(coords) < nd:
            coords.append(self.ir.Constant(i32, 0))
        coords = coords[:nd]
        cbrace = "{" + ",".join(f"${2 + i}" for i in range(nd)) + "}"
        # elect-one guard: UTMASTG is a uniform (once-per-warp) SASS op — see
        # _elect_one_wrap. commit_group stays per-thread (matches each
        # thread's own bulk-group state that wait_group 0 later drains).
        body = (f"cp.async.bulk.tensor.{nd}d.global.shared::cta.bulk_group "
                f"[$0, {cbrace}], [$1]")
        asm_str = self._elect_one_wrap(body) + " cp.async.bulk.commit_group;"
        cons = "l,l," + ",".join(["r"] * nd) + ",~{memory}"
        fn_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i64] + [i32] * nd)
        asm = self.ir.InlineAsm(fn_ty, asm_str, cons, side_effect=True)
        IRBuilder.call(asm, [desc_v, smem_ptr] + coords, "")

    @lift_for("UTMACCTL")
    def lift_utmacctl(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UTMACCTL.PF [URd] — tensormap prefetch hint. Safe NOP (prefetch is
        a performance hint; prefetch.tensormap needs the same pointer plumbing
        as UTMALDG and adds nothing to correctness)."""
        pass

    @lift_for("UBLKCP")
    def lift_ublkcp(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UBLKCP.S.G [URt],[URsrc],URsize -> cp.async.bulk (non-tensor TMA
        bulk copy global->shared). URt = {smem dst, mbarrier} tuple (parser
        synthesizes the mbar member, mirroring UTMALDG); URsrc:URsrc+1 = the
        64-bit global source; URsize = size in 16-BYTE UNITS. Uniform op ->
        elect-one wrap (see _elect_one_wrap). Completion rides the mbarrier
        complete_tx protocol like the tensor loads. FA3 bwd uses this for
        the LSE_log2/dPsum row loads (was a silent placeholder -> smem never
        filled -> the whole bwd pipeline deadlocked in try_wait)."""
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        mem_ops = [op for op in uses if op.is_mem_addr]
        if len(mem_ops) < 2:
            raise UnsupportedInstructionException(f"UBLKCP needs 2 mem ops: {Inst.opcodes}")
        dst_v = _get_val(mem_ops[0], "ublkcp_dst")
        dst_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, dst_v, "ublkcp_dst_gep"), i64, "ublkcp_dst_ptr")
        src_v = _get_val(mem_ops[1], "ublkcp_src")
        src_v = self._insert_type_cast_if_needed(IRBuilder, src_v, i64, "ublkcp_src64")
        mbar_v = None
        size_op = None
        for u in uses:
            if getattr(u, "tma_tuple_idx", None) == 1:
                mbar_v = _get_val(u, "ublkcp_mbar")
            elif u.is_reg and not u.is_mem_addr and \
                    getattr(u, "tma_tuple_idx", None) is None and \
                    getattr(u, "reg", None) not in (None, "RZ", "URZ"):
                size_op = u
        if mbar_v is None or size_op is None:
            raise UnsupportedInstructionException(f"UBLKCP mbar/size missing: {Inst.opcodes}")
        mbar_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, mbar_v, "ublkcp_mbar_gep"), i64, "ublkcp_mbar_ptr")
        units = _get_val(size_op, "ublkcp_units")
        units = self._insert_type_cast_if_needed(IRBuilder, units, i32, "ublkcp_units32")
        nbytes = IRBuilder.shl(units, self.ir.Constant(i32, 4), "ublkcp_bytes")
        body = ("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
                " [$0], [$1], $2, [$3]")
        fn_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i64, i32, i64])
        asm = self.ir.InlineAsm(
            fn_ty, self._elect_one_wrap(body), "l,l,r,l,~{memory}", side_effect=True)
        IRBuilder.call(asm, [dst_ptr, src_v, nbytes, mbar_ptr], "")

    @lift_for("UBLKRED")
    def lift_ublkred(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UBLKRED.G.S.ADD.F32.RN [URg],[URs],URsize,desc[URm] ->
        cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 (TMA bulk
        reduce-add global += shared) + commit_group. URg:URg+1 = 64-bit
        global dst; URs = shared src byte offset; URsize = 16-byte units;
        desc[URm] = access memdesc (ignored, like the TMA loads). Drained by
        the kernel's own ACQBULK -> cp.async.bulk.wait_group 0. Uniform op ->
        elect-one wrap. This is FA3 bwd's ONLY dQ_accum writer (was a silent
        placeholder -> dQ would come out all-zero)."""
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        mem_ops = [op for op in uses if op.is_mem_addr
                   and not getattr(op, "tma_memdesc", False)
                   and getattr(op, "desc_reg", None) is None]
        if len(mem_ops) < 2:
            raise UnsupportedInstructionException(f"UBLKRED needs 2 mem ops: {Inst.opcodes}")
        gmem_v = _get_val(mem_ops[0], "ublkred_gmem")
        gmem_v = self._insert_type_cast_if_needed(IRBuilder, gmem_v, i64, "ublkred_gmem64")
        smem_v = _get_val(mem_ops[1], "ublkred_smem")
        smem_ptr = IRBuilder.ptrtoint(
            self._shared_gep(IRBuilder, smem_v, "ublkred_smem_gep"), i64, "ublkred_smem_ptr")
        size_op = next(
            (u for u in uses if u.is_reg and not u.is_mem_addr
             and getattr(u, "tma_tuple_idx", None) is None
             and getattr(u, "desc_reg", None) is None
             and getattr(u, "reg", None) not in (None, "RZ", "URZ")),
            None)
        if size_op is None:
            raise UnsupportedInstructionException(f"UBLKRED size missing: {Inst.opcodes}")
        units = _get_val(size_op, "ublkred_units")
        units = self._insert_type_cast_if_needed(IRBuilder, units, i32, "ublkred_units32")
        nbytes = IRBuilder.shl(units, self.ir.Constant(i32, 4), "ublkred_bytes")
        body = ("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32"
                " [$0], [$1], $2")
        asm_str = self._elect_one_wrap(body) + " cp.async.bulk.commit_group;"
        fn_ty = self.ir.FunctionType(self.ir.VoidType(), [i64, i64, i32])
        asm = self.ir.InlineAsm(fn_ty, asm_str, "l,l,r,~{memory}", side_effect=True)
        IRBuilder.call(asm, [gmem_v, smem_ptr, nbytes], "")

    @lift_for("ELECT")
    def lift_elect(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """ELECT Pd, URd, mask -> elect.sync (one leader lane per warp).

        Writes the leader predicate; the elected-laneid UR dest (usually URZ)
        is dropped when URZ."""
        # PTX elect.sync REQUIRES membermask == the current active set (the
        # SASS op implicitly intersects with active lanes; PTX is UB -> hangs
        # when predicated-off lanes are named). Read %activemask at the site.
        i32 = self.ir.IntType(32)
        asm_ty = self.ir.FunctionType(i32, [])
        asm = self.ir.InlineAsm(
            asm_ty,
            "{.reg .pred p; .reg .b32 m; activemask.b32 m; "
            "elect.sync _|p, m; selp.u32 $0, 1, 0, p;}",
            "=r,~{memory}", side_effect=True)
        leader = IRBuilder.call(asm, [], "elect")
        pred = IRBuilder.icmp_unsigned(
            "!=", leader, self.ir.Constant(i32, 0), "elect_p")
        for d in Inst.get_defs():
            if d.is_predicate_reg and not d.is_pt:
                self._set_val(IRBuilder, IRRegs, d, pred)
                break

    @lift_for("USETMAXREG")
    def lift_usetmaxreg(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """USETMAXREG — warpgroup register reallocation (setmaxnreg).

        Purely an occupancy/register-budget hint from the producer/consumer
        warp specialization; correctness does not depend on it and emitting
        setmaxnreg constrains ptxas' own allocation of the LIFTED kernel
        (which has different register pressure). NOP. The TRY_ALLOC predicate
        dest (if any) is set TRUE so allocation-retry loops exit."""
        one = self.ir.Constant(self.ir.IntType(1), 1)
        for d in Inst.get_defs():
            if d.is_predicate_reg and not d.is_pt:
                self._set_val(IRBuilder, IRRegs, d, one)
                break

    @lift_for("ACQBULK")
    def lift_acqbulk(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """ACQBULK — acquire fence for prior bulk-async writes; map to
        cp.async.bulk.wait_group 0 (conservative: all groups drained)."""
        void_ty = self.ir.FunctionType(self.ir.VoidType(), [])
        asm = self.ir.InlineAsm(void_ty, "cp.async.bulk.wait_group 0;", "~{memory}",
                                side_effect=True)
        IRBuilder.call(asm, [], "")

    @lift_for("ENDCOLLECTIVE")
    def lift_endcollective(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """ENDCOLLECTIVE — scheduler hint closing a collective region; NOP."""
        pass

    @lift_for("WARPGROUP")
    def lift_warpgroup(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """WARPGROUP.ARRIVE / WARPGROUP.DEPBAR.LE -> wgmma fence/commit/wait.

        ptxas compiles the PTX wgmma bracket ops to this SASS pair:
          wgmma.fence.sync.aligned          -> WARPGROUP.ARRIVE
          wgmma.commit_group + wait_group N -> WARPGROUP.DEPBAR.LE gsbK, N
        Reverse them 1:1; the commit is folded into DEPBAR (SASS has no separate
        commit — group closure is implicit at the dependency barrier).
        """
        mods = [t.upper() for t in Inst.opcodes[1:]]
        void_ty = self.ir.FunctionType(self.ir.VoidType(), [])
        if "ARRIVE" in mods:
            asm = self.ir.InlineAsm(void_ty, "wgmma.fence.sync.aligned;", "~{memory}",
                                    side_effect=True)
            IRBuilder.call(asm, [], "")
            return
        if "DEPBAR" in mods:
            # SASS DEPBAR.LE gsbK, N waits on ONE of four wgmma scoreboards
            # (each HGMMA is tagged gsb0-3); PTX wait_group N counts a SINGLE
            # per-warpgroup commit queue. With several concurrent gemm chains
            # on different scoreboards (FA3 bwd hdim96: S/dP/dK/dQ), carrying
            # N over lets the group the SASS was waiting for stay in flight —
            # the dQ STS.128 read unretired accumulators and stored another
            # chain's stale register content (exact-values-at-wrong-places
            # dQaccum signature, dQ off by 8e-2). Waiting for ALL outstanding
            # groups (N=0) is the only sound flattening of per-scoreboard
            # waits onto the single PTX queue.
            asm = self.ir.InlineAsm(
                void_ty,
                "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0;",
                "~{memory}", side_effect=True)
            IRBuilder.call(asm, [], "")
            return
        # Other WARPGROUP forms: NOP.
