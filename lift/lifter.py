import contextlib
import inspect
import io
import os
import struct
import sys

import llvmlite
import llvmlite.binding as llvm
from llvmlite import binding, ir

from transform.transforms import Transforms
from transform.typeanalysis import (
    BOOL, FLOAT16, BF16, TF32, FLOAT32, FLOAT64,
    INT8, INT16, INT32, INT64, INT128,
    UINT8, UINT16, UINT32, UINT64,
)
from utils.elf import Elf


def lift_for(*opcodes):
    def decorator(func):
        func._lift_opcodes = tuple(opcodes)
        return func

    return decorator


class _NullWriter(io.TextIOBase):
    """Discards writes. Used to mute pass chatter without accumulating the
    output in memory the way redirect_stdout(StringIO) does — large kernels
    generate hundreds of MB of pass prints."""
    def write(self, s):
        return len(s)


class UnsupportedInstructionException(Exception):
    pass


class Lifter:
    NUM_CONST_MEM_BANKS = 5
    CONST_MEM_ADDRSPACE = 0
    # Hardware constant-bank window: reads past a section's data are legal in
    # SASS and return 0 on the GPU (see lifter_cpu._setup_const_mem padding).
    CONST_BANK_WINDOW = 65536
    # Backends whose warp-collective runtime identifies the *static*
    # collective by an explicit site argument (the SASS instruction address)
    # set this True. Keying by the binary call site (__builtin_return_address)
    # is NOT stable: clang -O2 tail-duplicates lifted blocks, so lanes
    # reaching the SAME logical SASS collective through different CFG paths
    # land on different clones — their divergence-history hashes split and
    # the warp group fractures (CB_strsm: every solve-loop SHFL.IDX returned
    # the caller's own value). The x86 runtime takes the site as a trailing
    # i32 instead; NVVM maps to llvm.nvvm.* intrinsics with fixed signatures
    # and must not receive it.
    collective_site_arg = False

    def __init__(self, elf, verbose=False, continue_on_error=False,
                 const_mem_linkage="external"):
        self._verbose = verbose
        # When True, skip any function whose lift raises (e.g. malformed
        # PHI nodes, type-inference disagreement, unhandled SASS) instead
        # of aborting the whole module. Used by best-effort multi-kernel
        # bulk-lift where one bad kernel shouldn't kill the cubin.
        self._continue_on_error = continue_on_error
        # `external` (default) preserves existing behavior: the host driver
        # in test/common/kernel_wrapper_*.h defines `const_mem` as a CPU
        # symbol that the lifted kernel references via the linker.
        # `internal` is for downstream pipelines that link multiple lifted
        # translation units into one binary — each TU's @const_mem is
        # private and the host marshalling lives elsewhere (e.g. PACT's
        # generated wrapper inside the IR). Avoids "multiple definition"
        # of @const_mem across TUs.
        if const_mem_linkage not in ("external", "internal"):
            raise ValueError(
                f"const_mem_linkage must be 'external' or 'internal', "
                f"got {const_mem_linkage!r}")
        self._const_mem_linkage = const_mem_linkage
        pkg_version = getattr(llvmlite, "__version__", None)
        llvm_ver = getattr(
            llvm,
            "llvm_version_info",
            getattr(llvm, "llvm_version", None),
        )

        if self._verbose:
            print("llvmlite package version:", pkg_version)
            if llvm_ver is not None:
                print("LLVM version string:", ".".join(map(str, llvm_ver)))
            print("")
            
        self.elf = elf
        self.arg_map = self.elf.arg_map
        self.elf_const_data = self.elf.const_data
        self.ir_functions = {}

        self.ir = ir
        self.lift_errors = []
        self.intrinsics = {}
        self.liftMap = self._build_lift_map()
        self.ConstMem = None
        self._placeholder_intrinsics = {}
        self.placeholder_log = []

    def _get_const_mem_bank_size(self, func_name):
        # const_mem uses a single uniform bank size across all banks, so it
        # must fit the LARGEST bank — including module-shared banks (e.g. the
        # nvjpeg IDCT zigzag / resize coefficient tables in bank 3, which can
        # be far larger than the per-kernel param bank). Sizing to the param
        # bank alone truncated those tables to zeros.
        cm_data = self.elf_const_data.get(func_name) or {}
        shared = getattr(self.elf, "shared_const_data", None) or {}
        max_size = 0
        for bank in range(self.NUM_CONST_MEM_BANKS):
            bank_size = 0
            if bank in cm_data:
                bank_size = max(bank_size, len(cm_data[bank]) * 4)
            if bank in shared:
                bank_size = max(bank_size, len(shared[bank]) * 4)
            max_size = max(max_size, bank_size)
        return max_size

    def _sp_base_value(self, func_name):
        """Value to materialise into c[0x0][0x28] (the per-thread stack-pointer
        base used by the SASS local-mem prologue `LDC R1, c[0x0][0x28]`).

        The driver supplies the SP base at launch; the SASS then chains
        `IADD3 R1, R1, -<frame>` per (sub)function and adds it back on return.
        After the lifter INLINES device-fn calls into one kernel, every inlined
        callee keeps its own carve, so R1 must start high enough to cover the
        DEEPEST point of the chain — i.e. EIATTR_MIN_STACK_SIZE, which is the
        kernel's own frame plus the deepest callee frame chain. Using only the
        kernel's EIATTR_FRAME_SIZE underflows R1 when the kernel's own frame is
        0 but callees carve locals (fresnel: frame=0, stack=0x1a0 -> R1 went
        negative -> out-of-bounds local write). For leaf kernels min_stack ==
        frame, so this is a safe superset. We take the max defensively."""
        frame_size = 0
        min_stack = 0
        if hasattr(self.elf, "frame_size"):
            frame_size = self.elf.frame_size.get(func_name, 0) or 0
        if hasattr(self.elf, "min_stack_size"):
            min_stack = self.elf.min_stack_size.get(func_name, 0) or 0
        return max(frame_size, min_stack)

    def _build_const_mem_initializer(self, func_name, bank_size):
        i8 = self.ir.IntType(8)
        bank_initializers = []
        # NVIDIA driver populates c[0x0][0x28] with the kernel's stack frame
        # size at launch (used by SASS code as the SP base — `LDC R1,
        # c[0x0][0x28]` followed by `VIADD R, R1, -frame_size` to compute
        # the start of the per-thread local-mem region). The cubin records
        # this as EIATTR_FRAME_SIZE; inject the value here so LDL/STL
        # addressing resolves to a valid LocalMem offset.
        frame_size = self._sp_base_value(func_name)
        for bank in range(self.NUM_CONST_MEM_BANKS):
            bank_bytes = [self.ir.Constant(i8, 0)] * bank_size
            words = self.elf.const_words(func_name, bank)
            if words:
                for word_idx, word in enumerate(words):
                    byte_offset = word_idx * 4
                    if byte_offset + 4 <= bank_size:
                        for b in range(4):
                            bank_bytes[byte_offset + b] = self.ir.Constant(
                                i8, (word >> (b * 8)) & 0xFF
                            )
            if bank == 0 and frame_size and bank_size >= 0x2C:
                for b in range(4):
                    bank_bytes[0x28 + b] = self.ir.Constant(
                        i8, (frame_size >> (b * 8)) & 0xFF
                    )
            bank_initializers.append(self.ir.Constant.literal_array(bank_bytes))
        return self.ir.Constant.literal_array(bank_initializers)

    def _setup_const_mem(self, llvm_module, func_name):
        if self.ConstMem is not None:
            return
        # Size to fit the LARGEST kernel in the module — multi-kernel cubins
        # share one @const_mem, so it must accommodate every kernel's arg
        # region. (For single-kernel modules this collapses to the same
        # size as before.)
        bank_size = self._max_const_mem_bank_size_in_module()
        if bank_size == 0:
            bank_size = self._get_const_mem_bank_size(func_name)
        if bank_size == 0:
            bank_size = 256
        bank_ty = self.ir.ArrayType(self.ir.IntType(8), bank_size)
        const_array_ty = self.ir.ArrayType(bank_ty, self.NUM_CONST_MEM_BANKS)
        self.ConstMem = self.ir.GlobalVariable(
            llvm_module, const_array_ty, "const_mem", self.CONST_MEM_ADDRSPACE
        )
        # Force 16-byte alignment so ptxas-emitted wide loads/stores against
        # const_mem (e.g. st.global.v2.b64 [const_mem+352]) don't fault.
        self.ConstMem.align = 16
        # Linkage is configurable. The default ('external' — handled here
        # by leaving llvmlite's linkage attribute unset, since the LLVM
        # IR text for default linkage on a defined global is just the
        # absence of any keyword) lets a host driver define and populate
        # `const_mem` from outside the lifted module — that's how
        # SLifter's own x86 test harnesses work. 'internal' makes each
        # TU's @const_mem private, suitable for downstream pipelines that
        # link many lifted TUs together (e.g. CuFuzz/PACT) and provide
        # arg marshalling inside the IR.
        if self._const_mem_linkage == "internal":
            self.ConstMem.linkage = "internal"
        # else: leave .linkage alone -> emits as default-linkage global.
        self.ConstMem.initializer = self._build_const_mem_initializer(func_name, bank_size)

    def _max_const_mem_bank_size_in_module(self):
        """Largest const-bank size across every kernel name we have data
        for. Returns 0 if elf_const_data is empty (e.g. binary cubin path
        where _parse_binary_nvinfo_sections is stubbed)."""
        if not self.elf_const_data:
            return 0
        max_size = 0
        for func_name in self.elf_const_data.keys():
            sz = self._get_const_mem_bank_size(func_name)
            if sz > max_size:
                max_size = sz
        return max_size

    def _should_load_const_from_memory(self, bank, byte_offset):
        return False

    def _load_const_from_elf_const_data(self, bank, byte_offset, target_ty):
        if self._should_load_const_from_memory(bank, byte_offset):
            return None
        is_float = target_ty.intrinsic_name[0] == 'f'
        width = int(target_ty.intrinsic_name[1:]) if is_float else target_ty.width
        num_bytes = width // 8
        func_name = self.curr_function.name

        # SASS prologue idiom: `LDC R1, c[0x0][0x28]` followed by
        # `VIADD R1, R1, -frame_size` initialises the per-thread stack
        # pointer. The driver populates c[0x0][0x28] with the frame size
        # at launch — the cubin records it as EIATTR_FRAME_SIZE. Inject
        # the value so LDL/STL addressing resolves to a valid LocalMem
        # offset instead of folding to 0.
        if (
            bank == 0
            and not is_float
            and byte_offset <= 0x28 < byte_offset + num_bytes
            and hasattr(self.elf, "frame_size")
        ):
            fs = self._sp_base_value(func_name)
            if fs:
                shift = (0x28 - byte_offset) * 8
                return self.ir.Constant(target_ty, fs << shift)

        bank_size = self.elf.const_size_bytes(func_name, bank)
        if byte_offset < 0 or byte_offset + num_bytes > bank_size:
            return None

        result = self.elf.read_const_int(func_name, bank, byte_offset, num_bytes, signed=False)
        if result is None:
            return None

        if is_float:
            if width == 32:
                result = struct.unpack('f', struct.pack('I', result & 0xFFFFFFFF))[0]
            elif width == 64:
                result = struct.unpack('d', struct.pack('Q', result & 0xFFFFFFFFFFFFFFFF))[0]

        return self.ir.Constant(target_ty, result)

    def _load_from_const_mem(self, builder, bank, byte_offset, target_ty, name=""):
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
        if byte_offset_val.type != self.ir.IntType(64):
            if byte_offset_val.type.width < 64:
                byte_offset_val = builder.zext(byte_offset_val, self.ir.IntType(64), f"{name}_off_zext")
            elif byte_offset_val.type.width > 64:
                byte_offset_val = builder.trunc(byte_offset_val, self.ir.IntType(64), f"{name}_off_trunc")

        ptr = builder.gep(
            self.ConstMem,
            [
                self.ir.Constant(self.ir.IntType(64), 0),
                self.ir.Constant(self.ir.IntType(64), bank),
                byte_offset_val,
            ],
            f"{name}_ptr" if name else "cm_dyn_ptr",
        )
        ptr = builder.bitcast(ptr, self.ir.PointerType(target_ty, self.CONST_MEM_ADDRSPACE))
        return builder.load(ptr, name if name else "cm_dyn_load")

    def _get_const_mem_val(self, builder, ir_regs, op, name=""):
        bank = op.const_mem_bank
        target_ty = op.get_ir_type(self)

        if op.is_reg and op.reg:
            reg_val = ir_regs[op.reg]
            if op.offset_value is not None:
                offset_const = self.ir.Constant(reg_val.type, int(op.offset_value))
                byte_offset_val = builder.add(reg_val, offset_const, f"{name}_cm_off")
            else:
                byte_offset_val = reg_val
            return self._load_from_const_mem_dynamic(builder, bank, byte_offset_val, target_ty, name)
        else:
            byte_offset = int(op.offset_value) if op.offset_value is not None else 0
            return self._load_from_const_mem(builder, bank, byte_offset, target_ty, name)

    def _resolve_descriptor_addr(self, Inst, ptr, addr, IRBuilder, IRRegs, _get_val):
        """Add descriptor base address for SM90 desc[URx][Ry.64] memory ops.

        The descriptor register (URx) holds a 64-bit base pointer loaded from
        const_mem. The register pair Ry.64 is an offset. The actual address
        is URx_base + Ry_offset.

        Hoisted to the base Lifter so BOTH backends inherit it (CpuLifter keeps
        its own byte-identical override; NVVMLifter now inherits this one so
        SM90 descriptor-addressed stores/loads — the FA3 fwd O epilogue
        `ST.E.128 desc[UR48][R14.64]` — resolve the real output base instead of
        writing to the bare offset). Returns `addr` UNCHANGED whenever no
        descriptor base is recoverable (no desc_reg, URZ, or an ambiguous/absent
        const map entry), so it is a no-op on every HeCBench / non-descriptor
        kernel and lifts stay byte-identical there.
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

    def _build_lift_map(self):
        lift_map = {}
        for cls in reversed(self.__class__.mro()):
            for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
                opcodes = getattr(method, "_lift_opcodes", None)
                if not opcodes:
                    continue
                bound_method = getattr(self, name)
                for opcode in opcodes:
                    lift_map[opcode] = bound_method
        return lift_map

    def lift_module(self, module, outfile):
        self.passes = self.get_transform_passes()

        # Optimization #2: deduplicate consecutive DefUseAnalysis runs.
        # Only keep the last one in each consecutive group.
        self.passes = self._dedup_defuse(self.passes)

        # Dynamic-shared detection MUST run on the RAW (pre-transform) module:
        # RegRemap collapses uniform registers (URx) and normal registers (Rx)
        # into one `R%d` namespace, erasing the uniform-vs-normal distinction.
        # The dynamic-shared discriminator (see NVVMLifter._detect_dynamic_shared)
        # depends on that distinction — a genuine dynamic-shared kernel addresses
        # shared through a UNIFORM-register base (`LDS/STS [URz+...]`) derived
        # from the CgaCtaId + UMOV(imm == static .nv.shared size) ULEA, whereas a
        # static kernel that merely reuses the same cluster constant routes it
        # through a normal register first. Capture the verdict here while the
        # uniform-register names still survive.
        detect = getattr(self, "_detect_dynamic_shared", None)
        if detect is not None:
            self._dynamic_shared_detected = detect(module)

        # Module-level pre-transforms run while every Function is still a flat,
        # untransformed instruction list AND all sibling Functions are visible
        # (the per-function pipeline below isolates each Function). Cross-
        # function CALL-ABI inlining lives here: it merges device-helper
        # Functions into their caller kernels so the existing intra-function
        # inline path can lower them. (Backends opt in via _pre_transform_module.)
        pre = getattr(self, "_pre_transform_module", None)
        if pre is not None:
            pre(module)

        # Optimization #1: per-function transform pipeline.
        # Each function is independent — run the full pipeline per function
        # instead of running each pass across ALL functions. This reduces
        # the working set from N*F instructions to N instructions (where
        # N = instructions per function, F = number of functions).
        self._transform_per_function(module)

        # Detect whether the kernel fills any SW128 GMMA operand via the
        # PackGQA cp.async gather (LDGSTS) rather than TMA. The gather's SASS
        # smem-dst applies an extra per-row XOR of ((row & 9) << 3) on the
        # element index (measured exact vs the runtime gather, 0-mismatch),
        # which the WGMMA SW128 A-read must mirror to land on the cells the
        # gather actually wrote. TMA-loaded kernels (the base FA fwd kernels)
        # have ZERO LDGSTS, so this stays off and they are unaffected.
        #
        # NOTE (generalization, 2026-07 sweep): the three gather flags below
        # (_q_gather_loaded / _kv_gather_loaded / _v_cpasync_gathered) are
        # MODULE-WIDE instruction-count heuristics. The invariant each one
        # approximates is per-operand dataflow: "which loader (cp.async LDGSTS
        # vs TMA UTMALDG) WRITES the smem region that a given WGMMA operand
        # READS". A static re-key on that invariant was attempted and
        # abandoned: both loaders' smem destinations are runtime register
        # bases (LDGSTS dst = [Rx+imm] where Rx carries region base + per-lane
        # swizzled offset; UTMALDG dst = [URx]), the WGMMA side reads via the
        # descriptor start_address (a runtime UR pair, stage-rotated), and in
        # a warp-specialized kernel producer and consumer compute those bases
        # in different warps — so region attribution is not statically
        # resolvable in general.
        # Known-good corpus for these thresholds:
        # the 63-kernel FA3 suite (fp16/bf16/e4m3 x fwd/bwd x base/paged/
        # packgqa/split/softcap variants) + test_sm90_bench (431 tests).
        _ldgsts_n = 0
        _utma_n = 0
        _ublkcp_n = 0
        for func in module.functions:
            for bb in func.blocks:
                for inst in bb.instructions:
                    op0 = inst.opcodes[0] if inst.opcodes else None
                    if op0 == "LDGSTS":
                        _ldgsts_n += 1
                    elif op0 == "UTMALDG":
                        _utma_n += 1
                    elif op0 == "UBLKCP":
                        _ublkcp_n += 1
        self._q_gather_loaded = _ldgsts_n > 0
        # A kernel that loads 1D tiles via UBLKCP (FA3 backward: the LSE/dPsum
        # softmax-recompute scalars) needs the producer's expect-tx mbarrier arrive
        # to DEFER its phase flip until those synchronous bulk copies publish to
        # smem — otherwise a consumer fiber, released by the eager flip, reads the
        # tile before the copy runs (P=exp2(...-0)=1.0 -> dV/dK/dQ garbage). Kernels
        # with no UBLKCP (every forward variant) keep the eager count-arrive so
        # their lifted IR is byte-identical. Consumed by lift_syncs (ARRIVE).
        self._has_ublkcp = _ublkcp_n > 0
        # Detect whether the K/V GMMA operands (the QK B=K and PV B=V tiles) are
        # filled by cp.async (PagedKVNonTMA) rather than TMA. PackGQA gathers only
        # Q (a small tile) while TMA-loading the bulk K/V, so its LDGSTS count is
        # always below its UTMALDG count; a PagedKVNonTMA kernel cp.async-gathers
        # K AND V, inverting that (LDGSTS >> UTMALDG). When set, the WGMMA B-reads
        # must mirror the same ((row & 9) << 3) cp.async smem swizzle as the A-read.
        self._kv_gather_loaded = _ldgsts_n > _utma_n
        # Stronger flag: V (and K) are cp.async-GATHERED into smem (PagedKVNonTMA),
        # NOT TMA-loaded. The coarse `_ldgsts_n > _utma_n` above wrongly fires for
        # the fp8 PackGQA kernels (which TMA-load V but gather only Q with FEWER TMA
        # loads than fp16, so LDGSTS just edges past UTMALDG — ratio ~1.7×). A true
        # PagedKVNonTMA kernel cp.async-gathers Q+K+V, so LDGSTS dwarfs UTMALDG (only
        # a handful of residual TMA ops remain — ratio ~49×). Use a wide ratio margin
        # so PackGQA (V via TMA) stays off while paged (V gathered) is on. Consumed by
        # the fp8 PV Vt B-read, which uses a DIFFERENT (cute-nominal) Vt layout when V
        # is gathered vs the TMA-V layout cf6f0322 decoded.
        self._v_cpasync_gathered = _ldgsts_n > 4 * _utma_n

        self.llvm_module = self.ir.Module(module.name)
        self.intrinsics = {}
        self.ir_functions = {}

        # Stash the parsed (post-transform) SIR module so setup_module can
        # inspect the SASS — e.g. to detect the dynamic-shared base idiom for
        # @shared_mem sizing. Minimal plumbing: setup_module otherwise only
        # receives the (empty) llvm_module.
        self.sir_module = module

        self.setup_module(self.llvm_module)

        self._declare_ir_functions(module.functions, self.llvm_module)

        skipped = []
        for func in module.functions:
            self.curr_function = func
            try:
                self._lift_function(func, self.llvm_module)
            except Exception as e:
                if not self._continue_on_error:
                    raise
                # Best-effort mode: drop the partially-built function so
                # the verifier doesn't trip later on a half-translated
                # body, log it, and continue with remaining kernels.
                ir_fn = self.ir_functions.pop(func.name, None)
                if ir_fn is not None:
                    try:
                        ir_fn.parent.functions.remove(ir_fn)
                    except (ValueError, AttributeError):
                        pass
                skipped.append((func.name, str(e)))

        if skipped:
            print(
                f"[lift_module] skipped {len(skipped)} function(s) under "
                f"--continue-on-error:",
                file=sys.stderr,
            )
            for name, err in skipped:
                short = err.splitlines()[0][:120] if err else ""
                print(f"  - {name}: {short}", file=sys.stderr)

        output_ir = str(self.llvm_module)

        output_ir = self.postprocess_ir(output_ir)

        if self._verbose:
            print(output_ir)
        print(output_ir, file=outfile)

        if self.placeholder_log:
            self._print_placeholder_summary()

    @staticmethod
    def _dedup_defuse(passes):
        """Remove redundant consecutive DefUseAnalysis passes.

        Before: [A, DefUse, DefUse, B, DefUse, C, DefUse, DefUse]
        After:  [A, DefUse, B, DefUse, C, DefUse]
        """
        from transform.defuse_analysis import DefUseAnalysis
        result = []
        for p in passes:
            if (isinstance(p, DefUseAnalysis) and result
                    and isinstance(result[-1], DefUseAnalysis)):
                continue  # skip duplicate
            result.append(p)
        removed = len(passes) - len(result)
        if removed:
            print(f"[opt] Removed {removed} redundant DefUseAnalysis pass(es)")
        return result

    def _transform_per_function(self, module):
        """Run the transform pipeline independently per function.

        This avoids O(F*N) scaling where F=functions, N=instructions.
        Each function sees only its own instructions during transforms.
        """
        from types import SimpleNamespace
        all_functions = list(module.functions)
        if len(all_functions) <= 1:
            # Single function — just run normally
            transforms = Transforms(self.passes, verbose=self._verbose)
            if not self._verbose:
                with contextlib.redirect_stdout(_NullWriter()):
                    transforms.apply(module)
            else:
                transforms.apply(module)
            return

        print(f"[opt] Per-function pipeline: {len(all_functions)} functions")
        transformed_functions = []
        for i, func in enumerate(all_functions):
            # Create a lightweight single-function wrapper
            single = SimpleNamespace(
                name=module.name,
                isa=module.isa,
                elf=module.elf,
                functions=[func],
            )
            # Fresh pass instances for each function (passes may have state)
            passes = self._fresh_passes()
            transforms = Transforms(passes, verbose=self._verbose)
            try:
                if not self._verbose:
                    with contextlib.redirect_stdout(_NullWriter()):
                        transforms.apply(single)
                else:
                    transforms.apply(single)
                # handle_call.py and similar transforms may APPEND extracted
                # device-callee functions to single.functions during the
                # pipeline (parent at index 0, extracted callees at 1+).
                # `extend` preserves all of them; the previous `append(
                # single.functions[0])` silently dropped extracted callees,
                # leaving CALLFUNC sites referring to undefined symbols and
                # producing PTX with unresolved externs that ptxas /
                # cuModuleLoad reject.
                transformed_functions.extend(single.functions)
                n_insts = sum(len(bb.instructions) for bb in single.functions[0].blocks)
                print(f"  [{i+1}/{len(all_functions)}] {func.name[:60]:60s} {n_insts:6d} insts")
            except Exception as e:
                print(f"  [{i+1}/{len(all_functions)}] {func.name[:60]:60s} ERROR: {e}")
                # Under CULIFTER_NO_FALLBACK (B3/B4 ablation) propagate per-
                # function transform failures so the caller counts the whole
                # cubin as failing to lift — otherwise the "partial" path
                # below silently recovers and the ablation underreports.
                if os.environ.get("CULIFTER_NO_FALLBACK") == "1":
                    raise
                # Still include function with partial transforms for best-effort lifting
                transformed_functions.append(func)
        module.functions = transformed_functions

    def _fresh_passes(self):
        """Create fresh pass instances (same types as self.passes)."""
        fresh = []
        for p in self.passes:
            cls = type(p)
            # TypeAnalysis may have max_iterations
            if hasattr(p, 'max_iterations'):
                fresh.append(cls(max_iterations=p.max_iterations))
            else:
                try:
                    fresh.append(cls())
                except TypeError:
                    fresh.append(p)  # reuse if can't construct
        return fresh

    def _print_placeholder_summary(self):
        from collections import Counter
        counts = Counter(entry["opcode"] for entry in self.placeholder_log)
        total = sum(counts.values())
        print(f"\n[SLifter] {total} unsupported instruction(s) replaced with placeholders:", flush=True)
        for opcode, count in counts.most_common():
            print(f"  {opcode}: {count}", flush=True)

    def postprocess_ir(self, ir_code):
        # Repair phi nodes whose incoming-block list doesn't match the
        # emitted CFG. The lifter occasionally diverges from the SSA pass
        # (placeholder ret void, R2P bypassing SSA, direct-CALL lifted as
        # branch, etc.) leaving phis that LLVM rejects with "PHINode should
        # have one entry for each predecessor of its parent basic block".
        # See lift/phi_repair.py.
        from lift.phi_repair import repair_stale_phi_entries
        return repair_stale_phi_entries(ir_code)

    def setup_module(self, llvm_module):
        raise NotImplementedError

    def run_intrinsic(self, name, builder, args, call_name=""):
        raise NotImplementedError

    def get_transform_passes(self):
        raise NotImplementedError

    def _get_ir_function_type(self, func):
        return self.ir.FunctionType(self.ir.VoidType(), [], False)

    def _declare_ir_functions(self, functions, llvm_module):
        for func in functions:
            if func.name in self.ir_functions:
                continue
            fn_ty = self._get_ir_function_type(func)
            self.ir_functions[func.name] = self.ir.Function(llvm_module, fn_ty, func.name)

    def _lift_basic_block(self, bb, builder, ir_regs, block_map):
        for inst in bb.instructions:
            try:
                self.lift_instruction(inst, builder, ir_regs, block_map)
            except UnsupportedInstructionException as e:
                opcode = inst.opcodes[0] if inst.opcodes else "UNKNOWN"
                self._emit_placeholder(builder, inst, opcode, ir_regs)
                func_name = getattr(self, 'curr_function', None)
                if func_name is not None:
                    func_name = getattr(func_name, 'name', str(func_name))
                self.placeholder_log.append({
                    "opcode": opcode,
                    "opcodes": list(inst.opcodes),
                    "inst_id": inst.id,
                    "function": func_name,
                    "error": str(e),
                })

    def _emit_placeholder(self, builder, inst, opcode, ir_regs=None):
        safe_name = opcode.replace(".", "_").lower()
        key = f"sass_unsupported_{safe_name}"
        if key not in self._placeholder_intrinsics:
            func_ty = self.ir.FunctionType(self.ir.VoidType(), [])
            # Use a plain identifier (no @llvm.* prefix) so downstream
            # toolchains can provide stub definitions. LLVM's module verifier
            # rejects defining functions in the reserved `@llvm.*` namespace,
            # which blocks building stub libraries through any LLVM-backed
            # pipeline (afl-clang-fast's -emit-llvm path, opt passes, etc).
            fn = self.ir.Function(
                self.llvm_module, func_ty,
                f"sass_unsupported_{safe_name}"
            )
            # Emit a weak no-op body so the resulting PTX is
            # self-contained: cuModuleLoad / ptxas reject unresolved
            # extern functions, which kills any standalone JIT path.
            # Weak linkage lets a downstream toolchain (PACT, CPU
            # stubs) provide a real body that overrides the no-op at
            # link time.
            fn.linkage = "weak"
            entry_bb = fn.append_basic_block("entry")
            self.ir.IRBuilder(entry_bb).ret_void()
            self._placeholder_intrinsics[key] = fn
        builder.call(self._placeholder_intrinsics[key], [], f"placeholder_{safe_name}")

        # Zero-fill def registers so that downstream instructions using
        # results of unsupported ops get a zero rather than KeyError.
        if ir_regs is not None:
            for def_op in inst.get_defs():
                self._zero_fill_def(builder, ir_regs, def_op)

        # CALL placeholders can end up in blocks with no other instructions;
        # LLVM requires every basic block to have a terminator. If the block
        # is currently terminator-less at the end of instruction lifting, the
        # block-lifting loop in lifter would drop us into a following block
        # via SSA fall-through that never gets emitted. Ensuring a terminator
        # is added at block-close is the block-loop's responsibility; here we
        # just make sure the placeholder is never the *last* thing we leave
        # in a terminator-less state if the SASS-side op was a control op.
        if opcode in ("CALL", "BRX", "RTS", "RET"):
            # Safe default: make this block return. CALL in SASS often encodes
            # tail-call-like device-function dispatch; for fuzzing purposes,
            # ret-void is a conservative no-op that keeps LLVM valid.
            try:
                builder.ret_void()
            except Exception:
                # If block already has a terminator or return type mismatch,
                # fall back to `unreachable`.
                try:
                    builder.unreachable()
                except Exception:
                    pass

    def _zero_fill_def(self, builder, ir_regs, operand):
        """Write a zero constant to ir_regs for an operand's register(s).

        Called when an unsupported instruction's def would otherwise
        leave ir_regs without an entry, causing KeyError on later uses.
        """
        if operand.is_reg_group:
            for sub_op in operand.reg_group:
                self._zero_fill_def(builder, ir_regs, sub_op)
            return

        if not operand.is_writable_reg:
            return

        ir_name = operand.get_ir_name(self)
        if ir_name is None:
            return
        # Don't overwrite a register that already has a value
        if ir_name in ir_regs:
            return

        # Use the operand's type if available, otherwise default to i32
        try:
            ty = operand.get_ir_type(self)
        except Exception:
            ty = self.ir.IntType(32)
        ir_regs[ir_name] = self.ir.Constant(ty, 0)

    def _populate_phi_nodes(self, bb, builder, ir_regs, block_map):
        ir_block = block_map[bb]

        for idx, inst in enumerate(bb.instructions):
            if inst.opcodes[0] not in {"PHI", "PHI64"}:
                continue

            ir_inst = ir_block.instructions[idx]
            # Use the PHI's actual IR type rather than the operand's declared
            # type — for predicate PHIs we widen i1→i32 in lift_phi (BUG-082)
            # and the incoming values must match the PHI's type, not the
            # SIR operand's type.
            target_ty = ir_inst.type

            for pred_idx, op in enumerate(inst.operands[1:]):
                pred_bb = bb.preds[pred_idx]

                if op.is_pt:
                    # int(...) — see _get_val: bare bool with non-i1 target
                    # type emits `i32 true` which llc rejects.
                    val = self.ir.Constant(target_ty, int(not op.is_not_reg))
                elif op.is_rz:
                    val = self.ir.Constant(target_ty, 0)
                elif op.is_writable_reg:
                    val = ir_regs.get(op.reg)
                    if val is None:
                        # Register not defined in predecessor — use zero.
                        # This can happen when NOP'd instructions (e.g., UTMALDG)
                        # should have produced side effects that define registers
                        # downstream (e.g., via shared memory loads).
                        val = self.ir.Constant(target_ty, 0)
                    elif op.is_predicate_reg \
                            and hasattr(val, "parent") and val.parent is not None \
                            and val.parent is not block_map[pred_bb] \
                            and getattr(val, "name", "").startswith("r2p_p"):
                        # R2P writes "P0..P6" by physical name, bypassing
                        # SSA. `ir_regs[Pn]` then returns the latest R2P
                        # write — its def's BB may not dominate this
                        # predecessor, so inserting a cast at pred_bb
                        # violates SSA dominance ("Instruction does not
                        # dominate all uses!"). Detect R2P outputs by name
                        # and substitute 0, matching lift_p2r's workaround.
                        # Legitimate SSA-controlled predicate flows (named
                        # via the SSA renamer, not "r2p_p*") fall through
                        # to the normal cast path.
                        val = self.ir.Constant(target_ty, 0)
                    else:
                        with builder.goto_block(block_map[pred_bb]):
                            val = self._insert_type_cast_if_needed(builder, val, target_ty, f"phi_cast_{op.name}")
                else:
                    raise UnsupportedInstructionException("not possible")
                ir_inst.add_incoming(val, block_map[pred_bb])

    def _ensure_block_terminators(self, func):
        # Unsupported control-flow ops (BSSY/BSYNC/BRX/BAR/BRA) emit only a
        # call to llvm.sass.unsupported.<op>. When such an op is the last
        # instruction in a block, the IR block lacks a terminator and the
        # module won't verify. Append a br to the first mapped successor, or
        # `unreachable` when no successor exists.
        for bb in func.blocks:
            ir_block = func.block_map.get(bb)
            if ir_block is None or ir_block.is_terminated:
                continue
            builder = self.ir.IRBuilder(ir_block)
            succ_ir = None
            for succ in bb.succs:
                candidate = func.block_map.get(succ)
                if candidate is not None:
                    succ_ir = candidate
                    break
            if succ_ir is not None:
                builder.branch(succ_ir)
            else:
                builder.unreachable()

    @staticmethod
    def _ty_kind_bits(intrinsic_name, ty):
        """Return ('i'|'f', bits) for a type's intrinsic_name.

        llvmlite emits intrinsic_name "i32" / "f32" / "f16" / "f64" for int
        and standard floats. Our BFloat16Type uses "bf16" — treat as float.
        """
        if intrinsic_name.startswith("bf"):
            return "f", int(intrinsic_name[2:])
        if intrinsic_name[0] == "i":
            return "i", ty.width
        if intrinsic_name[0] == "f":
            return "f", int(intrinsic_name[1:])
        raise ValueError(f"Unsupported intrinsic_name: {intrinsic_name}")

    def _entry_alloca(self, IRBuilder, ty, name):
        """Alloca in the function entry block (static, loop-safe).

        An alloca emitted at the current position would be dynamic when the
        instruction sits inside a loop (e.g. DMMA in a GEMM K-loop) and grow
        the stack on every iteration; entry-block allocas are static slots
        (LLVM treats any constant-size alloca in the entry block as static,
        regardless of its position within the block).

        MUST use goto_block on the SAME builder: a second IRBuilder inserting
        into a block the main builder is appending to desynchronizes the main
        builder's integer anchor — every later instruction then lands one slot
        early, beaching whatever was last at the block end after the eventual
        terminator (observed as a dangling cast after `ret void`).
        """
        entry = IRBuilder.function.entry_basic_block
        with IRBuilder.goto_block(entry):
            return IRBuilder.alloca(ty, name=name)

    def _insert_type_cast_if_needed(self, builder, val, target_ty, name):
        val_ty = val.type
        if val_ty == target_ty:
            return val

        if type(val_ty) is ir.VectorType or type(target_ty) is ir.VectorType:
            return builder.bitcast(val, target_ty, f"{name}_bitcast" if name else "reg_bitcast")

        val_kind, val_bits = self._ty_kind_bits(val_ty.intrinsic_name, val_ty)
        target_kind, target_bits = self._ty_kind_bits(target_ty.intrinsic_name, target_ty)

        if val_kind == "i" and target_kind == "i":
            if val_bits < target_bits:
                return builder.zext(val, target_ty, f"{name}_zext" if name else "reg_zext")
            if val_bits > target_bits:
                return builder.trunc(val, target_ty, f"{name}_trunc" if name else "reg_trunc")
            return val

        if val_kind == "f" and target_kind == "f":
            # Same-bit-width float-to-float (e.g. f16 <-> bf16) is a bitcast.
            if val_bits == target_bits:
                return builder.bitcast(val, target_ty,
                                       f"{name}_bitcast" if name else "reg_bitcast")
            if val_bits < target_bits:
                return builder.fpext(val, target_ty, f"{name}_fpext" if name else "reg_fpext")
            return builder.fptrunc(val, target_ty, f"{name}_fptrunc" if name else "reg_fptrunc")

        # Mixed int/float: bitcast with size adjustment
        if val_bits == target_bits:
            return builder.bitcast(val, target_ty, f"{name}_bitcast" if name else "reg_bitcast")

        i_val = self.ir.IntType(val_bits)
        i_tgt = self.ir.IntType(target_bits)
        if val_kind != "i":
            val = builder.bitcast(val, i_val, f"{name}_to_int" if name else "to_int")
        if val_bits < target_bits:
            val = builder.zext(val, i_tgt, f"{name}_zext" if name else "zext")
        elif val_bits > target_bits:
            val = builder.trunc(val, i_tgt, f"{name}_trunc" if name else "trunc")
        if target_kind != "i":
            val = builder.bitcast(val, target_ty, f"{name}_bitcast" if name else "bitcast")
        return val

    def _normalize_i1_in_curr_bb(self, IRBuilder, val, name):
        """Normalize a predicate value to i1 inside the current basic block.

        Used by lift_isetp / lift_fsetp / lift_fsetrp to combine an incoming
        predicate (val3, p_low) that came through `_get_val`. Two issues
        motivate this helper:

        1. Type widening — _get_val may return i32 (or wider) when the source
           pseudo-register was typed as int earlier in the pipeline (e.g.,
           constant-fold of !PT).
        2. SSA dominance (BUG-082) — when the i1 was defined in another basic
           block whose def does not dominate the current use site, llvmlite /
           the LLVM verifier rejects the use. Routing the value through an
           i32 carrier in the current block (zext + icmp != 0) is a no-op for
           well-formed SSA and a fix for the cross-BB case (the i32 carrier
           is itself fed by a PHI when needed by SSA).

        Returns an i1 Value defined in the current basic block.
        """
        i1 = self.ir.IntType(1)
        i32 = self.ir.IntType(32)
        if val.type == i1:
            # Re-thread through i32 so the resulting i1 is a freshly-emitted
            # instruction in the current BB. zext of a cross-BB i1 still
            # carries the dominance constraint, but combined with icmp != 0
            # it produces a NEW i1 in the current block whose only operand
            # is the (well-formed) zext above. The zext itself is in this BB
            # too, so its operand needs to dominate this BB — which it
            # already must (otherwise SSA would have inserted a PHI).
            widened = IRBuilder.zext(val, i32, f"{name}_to_i32")
            return IRBuilder.icmp_unsigned(
                "!=", widened, self.ir.Constant(i32, 0), f"{name}_to_i1"
            )
        if isinstance(val.type, self.ir.IntType):
            # Wider int (e.g., i32 from !PT constant fold) → != 0.
            zero = self.ir.Constant(val.type, 0)
            return IRBuilder.icmp_unsigned("!=", val, zero, f"{name}_to_i1")
        # Float/pointer/etc. — bitcast (size-mismatched bitcast will fail loudly).
        return IRBuilder.bitcast(val, i1, f"{name}_bitcast_i1")

    @lift_for("UP2UR")
    def lift_up2ur(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UP2UR URd, UPR, URs, mask — pack masked UNIFORM predicates into URd.

        Uniform-datapath analog of P2R: URd = URs | (UP<i> << i for mask bits).
        The parser appends the masked UP<i> registers as explicit uses (after
        [UPR, URs, mask]) so SSA resolves each one's reaching UISETP def; the
        opaque `UPR` placeholder itself carries no value. (The old handler
        cast the UPR placeholder itself — garbage; FA3 bwd hdim256's
        mbarrier-phase pack `UP2UR UR12, UPR, URZ, 0x1` lifted to junk and
        drove dQ to inf.)
        """
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i1 = self.ir.IntType(1)

        rs_op = uses[1] if len(uses) > 1 else None
        if rs_op is not None and not rs_op.is_rz:
            result = _get_val(rs_op, "up2ur_base")
            result = self._insert_type_cast_if_needed(
                IRBuilder, result, i32, "up2ur_base_i32")
        else:
            result = self.ir.Constant(i32, 0)

        mask_op = uses[2] if len(uses) > 2 else uses[-1]
        mask = mask_op.immediate_value if mask_op.is_immediate else 0

        appended = list(uses[3:])  # [UPR, URs, mask, <UP ops...>]
        ai = 0
        for bit in range(7):
            if mask & (1 << bit):
                pred_op = appended[ai] if ai < len(appended) else None
                ai += 1
                if pred_op is None:
                    continue
                pred_val = _get_val(pred_op, f"up2ur_up{bit}")
                if hasattr(pred_val.type, 'width') and pred_val.type.width != 1:
                    pred_val = IRBuilder.trunc(pred_val, i1, f"up2ur_up{bit}_i1")
                pred_i32 = IRBuilder.zext(pred_val, i32, f"up2ur_up{bit}_i32")
                shifted = IRBuilder.shl(
                    pred_i32, self.ir.Constant(i32, bit), f"up2ur_up{bit}_shl")
                result = IRBuilder.or_(result, shifted, f"up2ur_or_{bit}")

        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("MOV", "MOV64", "UMOV", "R2UR")
    def lift_mov(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "mov")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("MOV32I")
    def lift_mov32i(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        if not src.is_immediate:
            raise UnsupportedInstructionException(f"MOV32I expects immediate, got: {src}")
        val = self.ir.Constant(src.get_ir_type(self), src.immediate_value)
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("SETZERO")
    def lift_setzero(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        zero_val = self.ir.Constant(dest.get_ir_type(self), 0)
        self._set_val(IRBuilder, IRRegs, dest, zero_val)

    @lift_for("EXIT")
    def lift_exit(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # `@pred EXIT Pg` is a CONDITIONAL exit: exit iff Pg (the @pred split
        # already gated this block via PBRA), else fall through to the next
        # block. Without honoring Pg, sheath's `@!P0 EXIT P1` (exit iff
        # x<XMAX && x>=0) drops the x>=0 guard, so x<0 particles skip the
        # `alive=false` store and crash the next kernel's scatter (OOB).
        # BuildCFG adds the fall-through edge for the guard operand.
        ops = Inst.operands
        guard = ops[0] if (ops and ops[0].is_predicate_reg
                           and not ops[0].is_pt) else None
        fall_bb = None
        if guard is not None and block_map is not None:
            for succ in Inst.parent.succs:
                if succ in block_map:
                    fall_bb = block_map[succ]
                    break
        if guard is None or fall_bb is None or _get_val is None:
            IRBuilder.ret_void()
            return
        cond = _get_val(guard, "exit_guard")
        cond = self._normalize_i1_in_curr_bb(IRBuilder, cond, "exit_guard")
        ret_bb = IRBuilder.function.append_basic_block(f"exit_{Inst.id}")
        IRBuilder.cbranch(cond, ret_bb, fall_bb)
        with IRBuilder.goto_block(ret_bb):
            IRBuilder.ret_void()

    @lift_for("RET")
    def lift_ret(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        IRBuilder.ret_void()

    @lift_for("PHI", "PHI64")
    def lift_phi(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        phi_ty = dest.get_ir_type(self)
        # BUG-082: predicate (i1) values stored in IRRegs leak cross-BB i1
        # references. We coerce predicate stores through an i32 carrier in
        # _set_val; PHI nodes for predicates must therefore be i32 too so
        # that the carrier type is consistent across the function.
        i1 = self.ir.IntType(1)
        if phi_ty == i1 and dest.is_predicate_reg:
            phi_ty = self.ir.IntType(32)
        phi_val = IRBuilder.phi(phi_ty, "phi")
        self._set_val(IRBuilder, IRRegs, dest, phi_val)

    @lift_for("IMAD", "UIMAD")
    def lift_imad(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imad_lhs")
        v2 = _get_val(uses[1], "imad_rhs")
        v3 = _get_val(uses[2], "imad_addend")

        high = "HI" in Inst.opcodes
        wide = "WIDE" in Inst.opcodes

        cin = uses[3] if len(uses) > 3 and not high else None
        cout = Inst.get_defs()[1] if len(Inst.get_defs()) > 1 else None

        carry0 = None
        carry1 = None

        # IMAD.WIDE.U32 / IMAD.HI.U32 = unsigned; without .U32 = signed.
        # Signed IMAD.HI/WIDE: BOTH operands are sign-extended (standard signed
        # multiply-high), including an immediate Rb. cuobjdump already renders a
        # signed immediate WITH its sign (e.g. the signed-divide-by-7 magic
        # `-0x6db6db6d`), so sign-extending reproduces the hardware's signed
        # mul. (ptxas selects the *unsigned* opcode `.U32` when it wants the
        # raw-bit-pattern / unsigned-magic-folds-the-+n trick — so a SIGNED
        # opcode unambiguously means a signed extension. An earlier zext-the-
        # immediate path double-counted the +n on the constant-divisor pattern
        # `IMAD.HI Rd, n, -magic, {RZ:n}` where ptxas folds the +n into the
        # {Rc:Rc+1} pair addend, giving wrong quotients — IM_idiv7_magic etc.)
        is_unsigned = "U32" in Inst.opcodes

        if wide:
            if is_unsigned:
                v1_64 = IRBuilder.zext(v1, self.ir.IntType(64), "imad_lhs_64")
                v2_64 = IRBuilder.zext(v2, self.ir.IntType(64), "imad_rhs_64")
            else:
                v1_64 = IRBuilder.sext(v1, self.ir.IntType(64), "imad_lhs_64")
                v2_64 = IRBuilder.sext(v2, self.ir.IntType(64), "imad_rhs_64")
            prod = IRBuilder.mul(v1_64, v2_64, "imad_tmp_64")
            addend = self._insert_type_cast_if_needed(IRBuilder, v3, prod.type, "imad_addend_64")
            tmp = IRBuilder.add(prod, addend, "imad")
        elif high:
            if is_unsigned:
                v1_64 = IRBuilder.zext(v1, self.ir.IntType(64), "imad_lhs_64")
                v2_64 = IRBuilder.zext(v2, self.ir.IntType(64), "imad_rhs_64")
            else:
                v1_64 = IRBuilder.sext(v1, self.ir.IntType(64), "imad_lhs_64")
                v2_64 = IRBuilder.sext(v2, self.ir.IntType(64), "imad_rhs_64")
            prod64 = IRBuilder.mul(v1_64, v2_64, "imad_tmp_64")
            if uses[2].is_reg_group:
                # sm70+ register addend names the 64-bit pair {Rc+1:Rc}
                # (paired in sir/instruction.py), added to the FULL product
                # before the high-word extraction — same datapath as
                # IMAD.WIDE.  E.g. mont_prod32's u = (t + m*n) >> 32 is one
                # IMAD.HI.U32 R17, R8, R17, R12 with {R13:R12} = t.
                addend64 = self._insert_type_cast_if_needed(IRBuilder, v3, self.ir.IntType(64), "imad_addend_64")
                sum64 = IRBuilder.add(prod64, addend64, "imad_sum_64")
                sum_hi = IRBuilder.lshr(sum64, self.ir.Constant(self.ir.IntType(64), 32), "imad_tmp_hi")
                prod = IRBuilder.trunc(sum_hi, dest.get_ir_type(self), "imad_tmp_hi_trunc")
                tmp = prod
            else:
                # Non-pair addend (immediate/const-mem): 32-bit add after
                # extraction.
                prod_hi = IRBuilder.lshr(prod64, self.ir.Constant(self.ir.IntType(64), 32), "imad_tmp_hi")
                prod = IRBuilder.trunc(prod_hi, dest.get_ir_type(self), "imad_tmp_hi_trunc")
                addend = self._insert_type_cast_if_needed(IRBuilder, v3, dest.get_ir_type(self), "imad_addend")
                tmp = IRBuilder.add(prod, addend, "imad")
        else:
            prod = IRBuilder.mul(v1, v2, "imad_tmp")
            addend = self._insert_type_cast_if_needed(IRBuilder, v3, prod.type, "imad_addend")
            tmp = IRBuilder.add(prod, addend, "imad")

        if cout and not cout.is_pt:
            carry0 = IRBuilder.icmp_unsigned("<", tmp, prod, "imad_cout0")

        if cin is not None:
            cin_val = _get_val(cin, "imad_cin")
            cin_ext = IRBuilder.zext(cin_val, tmp.type, "imad_cin_ext")
            tmp2 = IRBuilder.add(tmp, cin_ext, "imad_with_cin")
            if cout and not cout.is_pt:
                carry1 = IRBuilder.icmp_unsigned("<", tmp2, tmp, "imad_cout1")
            tmp = tmp2

        self._set_val(IRBuilder, IRRegs, dest, tmp)

        if cout and not cout.is_pt:
            if carry1 is not None:
                self._set_val(IRBuilder, IRRegs, cout, IRBuilder.or_(carry0, carry1, "imad_cout"))
            else:
                self._set_val(IRBuilder, IRRegs, cout, carry0)

    @lift_for("IMAD64")
    def lift_imad64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imad_lhs")
        v1_64 = IRBuilder.zext(v1, self.ir.IntType(64), "imad_lhs_64")
        v2 = _get_val(uses[1], "imad_rhs")
        v2_64 = IRBuilder.zext(v2, self.ir.IntType(64), "imad_rhs_64")
        v3 = _get_val(uses[2], "imad_addend")

        tmp = IRBuilder.mul(v1_64, v2_64, "imad_tmp")
        tmp = IRBuilder.add(tmp, v3, "imad")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("SDIV")
    def lift_sdiv(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """SDIV pseudo-op emitted by IntDivRecognize for the Granlund-Möller
        magic-divide pattern. Its operands are the chain's own unsigned
        MAGNITUDES (|dividend|, |divisor| — the IABS outputs for a signed C
        divide), and the leaf it replaces computes `hi(magic*|n|)` ~= the
        unsigned quotient, so emit `udiv` — but guard against
        `divisor == 0`. The original SASS
        chain (MUFU.RCP → IMAD.HI → correction) silently returns garbage
        when the runtime divisor is 0; x86 `udiv` traps with SIGFPE. ptxas
        never emits the chain from a guaranteed divide-by-zero, but
        auto-generated test drivers can pass 0 in a const-memory slot
        (e.g. `flip-cuda`'s `strides_contiguous` buffer once one element is
        zero). Match the hardware's "wrong but finite" behaviour with a
        select rather than crashing before the kernel can produce output."""
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = _get_val(uses[0], "sdiv_dividend")
        v2 = _get_val(uses[1], "sdiv_divisor")
        v1 = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "sdiv_dividend")
        v2 = self._insert_type_cast_if_needed(IRBuilder, v2, i32, "sdiv_divisor")
        is_zero = IRBuilder.icmp_unsigned("==", v2, self.ir.Constant(i32, 0), "sdiv_div_zero")
        safe_div = IRBuilder.select(is_zero, self.ir.Constant(i32, 1), v2, "sdiv_safe_div")
        result = IRBuilder.udiv(v1, safe_div, "sdiv")
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("UREM64", "UREM64HI", "UDIV64", "UDIV64HI")
    def lift_urem64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """UREM64/UREM64HI and UDIV64/UDIV64HI pseudo-ops emitted by
        IntDivRecognize for ptxas's software u64 divide/modulo chain
        (I2F.U64.RP → MUFU.RCP → IADD3 +0x1FFFFFFE → F2I.U64.TRUNC →
        Newton-Raphson). UREM64* replace the pre-correction REMAINDER pair;
        UDIV64* replace the pre-correction QUOTIENT pair (45_dual_gemm's
        coord=index/N divide). Each computes one 32-bit word of
        `(Dhi:Dlo) {urem|udiv} (dhi:dlo)` from four 32-bit word operands
        [Dlo, Dhi, dlo, dhi]. Same div-by-zero policy as SDIV: the SASS chain
        produces a garbage-but-finite value when the divisor is 0 (the chain's
        own tail SEL then substitutes 0xFFFFFFFF), while urem/udiv would trap —
        so select divisor 1 instead of crashing."""
        op = Inst.opcodes[0]
        is_div = op.startswith("UDIV64")
        is_hi = op.endswith("HI")
        tag0 = "udiv64" if is_div else "urem64"
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        words = []
        for idx, tag in enumerate(("dividend_lo", "dividend_hi",
                                   "divisor_lo", "divisor_hi")):
            v = _get_val(uses[idx], f"{tag0}_{tag}")
            v = self._insert_type_cast_if_needed(IRBuilder, v, i32, f"{tag0}_{tag}")
            words.append(IRBuilder.zext(v, i64, f"{tag0}_{tag}_64"))
        dividend = IRBuilder.or_(
            words[0],
            IRBuilder.shl(words[1], self.ir.Constant(i64, 32), f"{tag0}_dividend_hi_shl"),
            f"{tag0}_dividend")
        divisor = IRBuilder.or_(
            words[2],
            IRBuilder.shl(words[3], self.ir.Constant(i64, 32), f"{tag0}_divisor_hi_shl"),
            f"{tag0}_divisor")
        is_zero = IRBuilder.icmp_unsigned(
            "==", divisor, self.ir.Constant(i64, 0), f"{tag0}_div_zero")
        safe_div = IRBuilder.select(
            is_zero, self.ir.Constant(i64, 1), divisor, f"{tag0}_safe_div")
        if is_div:
            res = IRBuilder.udiv(dividend, safe_div, tag0)
        else:
            res = IRBuilder.urem(dividend, safe_div, tag0)
        if is_hi:
            res = IRBuilder.lshr(res, self.ir.Constant(i64, 32), f"{tag0}_hi")
        result = IRBuilder.trunc(res, i32, f"{tag0}_word")
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("FADD", "UFADD")
    def lift_fadd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f32 = self.ir.FloatType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "fadd_lhs"), f32, "fadd_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "fadd_rhs"), f32, "fadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fadd(v1, v2, "fadd"))

    @lift_for("FFMA", "UFFMA")
    def lift_ffma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f32 = self.ir.FloatType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "ffma_lhs"), f32, "ffma_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "ffma_rhs"), f32, "ffma_rhs")
        v3 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[2], "ffma_addend"), f32, "ffma_addend")
        # SASS FFMA is a fused multiply-add (single rounding); emit llvm.fma.f32,
        # not fmul+fadd (which rounds twice). Modifiers: .RM/.RP/.RZ select the
        # rounding intrinsic; .SAT clamps to [0,1] via llvm.minnum/maxnum (which
        # PTXAS lowers to FMNMX, unlike a select chain).
        mods = Inst.opcodes[1:] if len(Inst.opcodes) > 1 else []
        if "RM" in mods:
            tmp = self.run_intrinsic("fma_rm_f32", IRBuilder, [v1, v2, v3], "ffma_rm")
        elif "RP" in mods:
            tmp = self.run_intrinsic("fma_rp_f32", IRBuilder, [v1, v2, v3], "ffma_rp")
        elif "RZ" in mods:
            tmp = self.run_intrinsic("fma_rz_f32", IRBuilder, [v1, v2, v3], "ffma_rz")
        else:
            fma_fn = self.llvm_module.globals.get("llvm.fma.f32")
            if fma_fn is None:
                fma_fn = self.ir.Function(
                    self.llvm_module, self.ir.FunctionType(f32, [f32, f32, f32], False),
                    "llvm.fma.f32")
            tmp = IRBuilder.call(fma_fn, [v1, v2, v3], "ffma")
        if "SAT" in mods:
            # FFMA.SAT clamps result to [0.0, 1.0]. Use llvm.minnum/maxnum
            # so PTXAS lowers to FMNMX (NaN-safe min/max), not select chains.
            zero = self.ir.Constant(f32, 0.0)
            one = self.ir.Constant(f32, 1.0)
            minnum_fn = self.llvm_module.globals.get("llvm.minnum.f32")
            if minnum_fn is None:
                minnum_fn = self.ir.Function(
                    self.llvm_module, self.ir.FunctionType(f32, [f32, f32], False),
                    "llvm.minnum.f32")
            maxnum_fn = self.llvm_module.globals.get("llvm.maxnum.f32")
            if maxnum_fn is None:
                maxnum_fn = self.ir.Function(
                    self.llvm_module, self.ir.FunctionType(f32, [f32, f32], False),
                    "llvm.maxnum.f32")
            tmp = IRBuilder.call(minnum_fn, [tmp, one], "ffma_sat_hi")
            tmp = IRBuilder.call(maxnum_fn, [tmp, zero], "ffma_sat_lo")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("FMUL", "UFMUL")
    def lift_fmul(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f32 = self.ir.FloatType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "fmul_lhs"), f32, "fmul_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "fmul_rhs"), f32, "fmul_rhs")
        res = IRBuilder.fmul(v1, v2, "fmul")
        # .D2/.D4/.D8 result divisor: FMUL.D2 halves the product (used e.g. for the
        # 4-pixel 4:2:0 chroma average). Without this the result is 2x/4x too large.
        for _m in Inst.opcodes:
            if _m in ("D2", "D4", "D8"):
                res = IRBuilder.fmul(res, self.ir.Constant(f32, {"D2": 0.5, "D4": 0.25, "D8": 0.125}[_m]),
                                     "fmul_div")
                break
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("FMNMX")
    def lift_fmnmx(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # SASS FMNMX: PT → min(Ra,Rb), !PT → max(Ra,Rb).
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f32 = self.ir.FloatType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "fmnmx_lhs"), f32, "fmnmx_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "fmnmx_rhs"), f32, "fmnmx_rhs")
        pred = Inst.get_uses()[2]
        if pred.is_pt and pred.is_not_reg:
            # !PT → MAX: pick v1 when v1>v2
            r = IRBuilder.fcmp_ordered(">", v1, v2, "fmnmx")
        else:
            # PT or runtime-truthy → MIN: pick v1 when v1<v2
            r = IRBuilder.fcmp_ordered("<", v1, v2, "fmnmx")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.select(r, v1, v2, "fmnmx_select"))

    @lift_for("FMNMX3")
    def lift_fmnmx3(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """3-operand float min/max (SM100+).
        FMNMX3 Rd, Ra, Rb, Rc, PT  → Rd = min(Ra, Rb, Rc)
        FMNMX3 Rd, Ra, Rb, Rc, !PT → Rd = max(Ra, Rb, Rc)
        """
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        va = _get_val(uses[0], "fmnmx3_a")
        vb = _get_val(uses[1], "fmnmx3_b")
        vc = _get_val(uses[2], "fmnmx3_c")
        pred = uses[3]
        if pred.is_pt:
            # min(a, b, c) = min(min(a, b), c)
            cmp1 = IRBuilder.fcmp_ordered(">", va, vb, "fmnmx3_ab")
            tmp = IRBuilder.select(cmp1, vb, va, "fmnmx3_min_ab")
            cmp2 = IRBuilder.fcmp_ordered(">", tmp, vc, "fmnmx3_abc")
            result = IRBuilder.select(cmp2, vc, tmp, "fmnmx3_min")
        else:
            # max(a, b, c) = max(max(a, b), c)
            cmp1 = IRBuilder.fcmp_ordered("<", va, vb, "fmnmx3_ab")
            tmp = IRBuilder.select(cmp1, vb, va, "fmnmx3_max_ab")
            cmp2 = IRBuilder.fcmp_ordered("<", tmp, vc, "fmnmx3_abc")
            result = IRBuilder.select(cmp2, vc, tmp, "fmnmx3_max")
        self._set_val(IRBuilder, IRRegs, dest, result)

    # --- packed video-SIMD helpers (VIMNMX/VIADDMNMX/VIMNMX3 .S16/.U16/.S8/.U8) ---
    # These ops operate per packed lane (2×16 or 4×8), NOT on the full 32-bit
    # word. Lifting them scalar gave the wrong result for __vmax2/__vadd2-class
    # intrinsics (intrinsics-simd: VIMNMX.S16 lifted scalar -> vmaxs2 wrong).
    def _vsimd_lane_bits(self, opcodes):
        # Packed type modifier is e.g. "S16x2" (2 lanes of 16-bit), "U8x4"
        # (4 lanes of 8-bit). Plain "S32"/"U32" (or none) = scalar 32-bit.
        for o in opcodes[1:]:
            if "16x2" in o:
                return 16
            if "8x4" in o:
                return 8
        return 32

    def _vsimd_unsigned(self, opcodes):
        # A U-prefixed TYPE modifier (U16x2/U8x4/U32) -> unsigned. Skip the
        # mnemonic itself (UVIMNMX starts with U but isn't a type tag).
        return any(o.startswith("U") for o in opcodes[1:])

    def _vsimd_lanes(self, IRBuilder, v, lane_bits, name):
        """Split a 32-bit word into 32//lane_bits lane values (low lane first)."""
        i32 = self.ir.IntType(32)
        lane_ty = self.ir.IntType(lane_bits)
        out = []
        for k in range(32 // lane_bits):
            sh = k * lane_bits
            x = (IRBuilder.lshr(v, self.ir.Constant(i32, sh), f"{name}_{k}s")
                 if sh else v)
            out.append(IRBuilder.trunc(x, lane_ty, f"{name}_{k}"))
        return out

    def _vsimd_pack(self, IRBuilder, lanes, lane_bits, name):
        i32 = self.ir.IntType(32)
        acc = None
        for k, m in enumerate(lanes):
            m32 = IRBuilder.zext(m, i32, f"{name}_z{k}")
            sh = k * lane_bits
            if sh:
                m32 = IRBuilder.shl(m32, self.ir.Constant(i32, sh), f"{name}_sh{k}")
            acc = m32 if acc is None else IRBuilder.or_(acc, m32, f"{name}_or{k}")
        return acc

    def _vsimd_minmax(self, IRBuilder, v1, v2, lane_bits, is_unsigned, is_min, name):
        if lane_bits >= 32:
            cmpf = IRBuilder.icmp_unsigned if is_unsigned else IRBuilder.icmp_signed
            cmp = cmpf("<" if is_min else ">", v1, v2, f"{name}_cmp")
            return IRBuilder.select(cmp, v1, v2, name)
        a = self._vsimd_lanes(IRBuilder, v1, lane_bits, f"{name}_a")
        b = self._vsimd_lanes(IRBuilder, v2, lane_bits, f"{name}_b")
        cmpf = IRBuilder.icmp_unsigned if is_unsigned else IRBuilder.icmp_signed
        outs = []
        for k in range(len(a)):
            cmp = cmpf("<" if is_min else ">", a[k], b[k], f"{name}_c{k}")
            outs.append(IRBuilder.select(cmp, a[k], b[k], f"{name}_m{k}"))
        return self._vsimd_pack(IRBuilder, outs, lane_bits, name)

    def _vsimd_add(self, IRBuilder, v1, v2, lane_bits, name):
        if lane_bits >= 32:
            return IRBuilder.add(v1, v2, name)
        a = self._vsimd_lanes(IRBuilder, v1, lane_bits, f"{name}_a")
        b = self._vsimd_lanes(IRBuilder, v2, lane_bits, f"{name}_b")
        outs = [IRBuilder.add(a[k], b[k], f"{name}_s{k}") for k in range(len(a))]
        return self._vsimd_pack(IRBuilder, outs, lane_bits, name)

    def _vsimd_relu(self, IRBuilder, val, opcodes, name):
        """`.RELU` clamps each (signed) lane to >= 0 — per-lane max(x, 0)."""
        if "RELU" not in opcodes:
            return val
        lane_bits = self._vsimd_lane_bits(opcodes)
        zero = self.ir.Constant(self.ir.IntType(32), 0)
        return self._vsimd_minmax(IRBuilder, val, zero, lane_bits, False, False, f"{name}_relu")

    @lift_for("VIMNMX", "UVIMNMX")
    def lift_vimnmx(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "vimnmx_a"), i32, "vimnmx_a")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "vimnmx_b"), i32, "vimnmx_b")
        pred = uses[2]
        is_min = pred.is_pt and not pred.is_not_reg
        val = self._vsimd_minmax(
            IRBuilder, v1, v2, self._vsimd_lane_bits(Inst.opcodes),
            self._vsimd_unsigned(Inst.opcodes), is_min, "vimnmx")
        val = self._vsimd_relu(IRBuilder, val, Inst.opcodes, "vimnmx")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("VIADDMNMX")
    def lift_viaddmnmx(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "viaddmnmx_a"), i32, "viaddmnmx_a")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "viaddmnmx_b"), i32, "viaddmnmx_b")
        v3 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[2], "viaddmnmx_c"), i32, "viaddmnmx_c")
        # VIADDMNMX: per-lane add then per-lane min/max (packed for .S16/.U16).
        lane_bits = self._vsimd_lane_bits(Inst.opcodes)
        is_unsigned = self._vsimd_unsigned(Inst.opcodes)
        sum_val = self._vsimd_add(IRBuilder, v1, v2, lane_bits, "viaddmnmx_sum")
        pred = uses[3] if len(uses) > 3 else None
        is_min = bool(pred and pred.is_pt and not pred.is_not_reg)
        val = self._vsimd_minmax(IRBuilder, sum_val, v3, lane_bits, is_unsigned,
                                 is_min, "viaddmnmx")
        val = self._vsimd_relu(IRBuilder, val, Inst.opcodes, "viaddmnmx")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("REDUX")
    def lift_redux(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # REDUX.SUM.S32 UR4, R10 — warp-level reduction to uniform register.
        # Base fallback: identity (result = input value). CPU and NVVM
        # backends override this with target-specific reductions.
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "redux_src")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("FCHK")
    def lift_fchk(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # FCHK Px, num, denom — sets Px when the fast reciprocal-NR f32-divide
        # path is INVALID so the `@!Px BRA` falls through to the accurate
        # __internal_fdiv slow path (a CALL). Previously stubbed to constant 0,
        # so the fast path was ALWAYS taken: for an overflowed denominator
        # (e.g. ex2.approx(expf(250))=+inf) rcp.approx(inf)=0 then fma(-inf,0,1)
        # = NaN escaped (cobahh RSME=nan). Compute the real predicate: Px = denom
        # is 0/subnormal/inf/nan OR num is inf/nan. Test sign-masked f32 bits so
        # source -/|.| modifiers don't matter (uses[0]=num, uses[1]=denom).
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)

        def _f32_bits(op, tag):
            v = _get_val(op, tag)
            if isinstance(v.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                return IRBuilder.bitcast(v, i32, tag + "_bits")
            if v.type != i32:
                return self._insert_type_cast_if_needed(IRBuilder, v, i32, tag + "_i32")
            return v

        num = _f32_bits(uses[0], "fchk_num")
        den = _f32_bits(uses[1], "fchk_den")
        mask = self.ir.Constant(i32, 0x7FFFFFFF)
        inf = self.ir.Constant(i32, 0x7F800000)
        minnorm = self.ir.Constant(i32, 0x00800000)
        aden = IRBuilder.and_(den, mask, "fchk_aden")
        anum = IRBuilder.and_(num, mask, "fchk_anum")
        den_sub = IRBuilder.icmp_unsigned("<", aden, minnorm, "fchk_den_subz")
        den_inf = IRBuilder.icmp_unsigned(">=", aden, inf, "fchk_den_infnan")
        num_inf = IRBuilder.icmp_unsigned(">=", anum, inf, "fchk_num_infnan")
        px = IRBuilder.or_(
            IRBuilder.or_(den_sub, den_inf, "fchk_den_bad"), num_inf, "fchk_px")
        self._set_val(IRBuilder, IRRegs, dest, px)
        
    @lift_for("FSET")
    def lift_fset(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        
        v1 = _get_val(uses[0], "fset_lhs")
        v2 = _get_val(uses[1], "fset_rhs")
        pred = _get_val(uses[2], "fset_pred")
        
        bool_as_float = "BF" in Inst.opcodes
        cmp1 = Inst.opcodes[2]
        cmp2 = Inst.opcodes[-1]
        
        funcs_map = {
            "EQ": "==",
            "NE": "!=",
            "LT": "<",
            "LE": "<=",
            "GT": ">",
            "GE": ">=",
            "NEU": "!=",
            "LTU": "<",
            "LEU": "<=",
            "GTU": ">",
            "GEU": ">=",
            "EQU": "==",
            "MAX": "ogt",
            "MIN": "olt",
            "NAN": "uno",
            "NUM": "ord",
        }

        cmp2_map = {
            "AND": IRBuilder.and_,
            "OR": IRBuilder.or_,
            "XOR": IRBuilder.xor,
        }

        if cmp1 not in funcs_map:
            raise UnsupportedInstructionException(f"Unsupported FSET comparison: {cmp1}")

        isUnordered = "U" in cmp1
        if isUnordered:
            temp = IRBuilder.fcmp_unordered(funcs_map[cmp1], v1, v2, "fset_cmp1")
        else:
            temp = IRBuilder.fcmp_ordered(funcs_map[cmp1], v1, v2, "fset_cmp1")

        
        if cmp2 not in cmp2_map:
            raise UnsupportedInstructionException(f"Unsupported FSET comparison combiner: {cmp2}")
        
        temp = cmp2_map[cmp2](temp, pred, "fset_final")
        
        if bool_as_float:
            tval = ir.Constant(ir.FloatType(), 1.0)
            fval = ir.Constant(ir.FloatType(), 0.0)
        else:
            tval = ir.Constant(ir.IntType(32), 0xffffffff)
            fval = ir.Constant(ir.IntType(32), 0)
            
        temp = IRBuilder.select(temp, tval, fval)
                                   
        self._set_val(IRBuilder, IRRegs, dest, temp)
        
        

    @lift_for("DADD")
    def lift_dadd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f64 = self.ir.DoubleType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "dadd_lhs"), f64, "dadd_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "dadd_rhs"), f64, "dadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fadd(v1, v2, "dadd"))

    @lift_for("DFMA")
    def lift_dfma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # SASS DFMA is a fused multiply-add (single rounding); emit the fused
        # llvm.fma.f64 to match, not fmul+fadd (which rounds twice). Mirrors
        # lift_ffma's use of llvm.fma.f32. DFMA.RP/.RM/.RZ rounding modifiers
        # are not yet distinguished (default round-nearest).
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f64 = self.ir.DoubleType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "dfma_lhs"), f64, "dfma_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "dfma_rhs"), f64, "dfma_rhs")
        v3 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[2], "dfma_addend"), f64, "dfma_addend")
        fma_fn = self.llvm_module.globals.get("llvm.fma.f64")
        if fma_fn is None:
            fma_fn = self.ir.Function(
                self.llvm_module, self.ir.FunctionType(f64, [f64, f64, f64], False),
                "llvm.fma.f64")
        tmp = IRBuilder.call(fma_fn, [v1, v2, v3], "dfma")
        self._set_val(IRBuilder, IRRegs, dest, tmp)

    @lift_for("DMUL")
    def lift_dmul(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        f64 = self.ir.DoubleType()
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "dmul_lhs"), f64, "dmul_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "dmul_rhs"), f64, "dmul_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.fmul(v1, v2, "dmul"))

    @lift_for("ISCADD")
    def lift_iscadd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iscadd_lhs")
        v2 = _get_val(uses[1], "iscadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iscadd"))

    def _iadd3_get_operand(self, IRBuilder, _get_val, op, name):
        """Get an IADD3 operand in 64-bit, handling neg/not in 64-bit domain
        so that carry propagation is correct.

        Hardware IADD3 with `-src` computes ~src + 1 (two's complement negate)
        as part of the 33-bit sum.  If we negate at 32-bit and then zext, the
        carry from ~0+1 = 0x1_0000_0000 is lost.  Instead, strip the modifier,
        zext the raw 32-bit value to 64-bit, and apply the modifier there."""
        has_neg = getattr(op, 'is_negative_reg', False) and op.is_negative_reg
        has_not = getattr(op, 'is_not_reg', False) and op.is_not_reg
        i64 = self.ir.IntType(64)

        if has_neg or has_not:
            # Temporarily strip the prefix so _get_val returns the raw value
            orig_prefix = op.prefix
            op.prefix = None
            try:
                raw = _get_val(op, name)
            finally:
                op.prefix = orig_prefix
            raw_64 = IRBuilder.zext(raw, i64, f"{name}_raw64")
            # Bitwise NOT in 64-bit, masked to 32 bits
            not_64 = IRBuilder.not_(raw_64, f"{name}_not64")
            mask = self.ir.Constant(i64, 0xFFFFFFFF)
            not_masked = IRBuilder.and_(not_64, mask, f"{name}_not64_masked")
            if has_neg:
                # Negate = ~val + 1 in 64-bit: the +1 can overflow 32 bits,
                # producing a carry that IADD3 captures.
                one = self.ir.Constant(i64, 1)
                return IRBuilder.add(not_masked, one, f"{name}_neg64")
            else:
                return not_masked
        else:
            v = _get_val(op, name)
            # Ensure integer type before zext (value may be float due to type analysis)
            if hasattr(v.type, 'width') and v.type.width == 32:
                return IRBuilder.zext(v, i64, f"{name}_64")
            elif str(v.type) == 'float':
                v_i32 = IRBuilder.bitcast(v, self.ir.IntType(32), f"{name}_bitcast")
                return IRBuilder.zext(v_i32, i64, f"{name}_64")
            else:
                v_cast = self._insert_type_cast_if_needed(IRBuilder, v, self.ir.IntType(32), f"{name}_toi32")
                return IRBuilder.zext(v_cast, i64, f"{name}_64")

    def _iadd3_operand_narrow(self, IRBuilder, _get_val, op, ty, name):
        """Get an IADD3 operand at the destination width `ty` (no i64 widening),
        applying neg/not in that width. Valid only when no carry-OUT is read:
        the low-`ty.width` bits of the sum are identical whether computed at
        width `ty` or in i64 (`-src` == `~src+1` mod 2**width)."""
        has_neg = getattr(op, 'is_negative_reg', False) and op.is_negative_reg
        has_not = getattr(op, 'is_not_reg', False) and op.is_not_reg
        if has_neg or has_not:
            orig_prefix = op.prefix
            op.prefix = None
            raw = _get_val(op, name)
            op.prefix = orig_prefix
            raw = self._iadd3_coerce_int(IRBuilder, raw, ty, name)
            notv = IRBuilder.not_(raw, f"{name}_not")
            if has_neg:
                return IRBuilder.add(notv, self.ir.Constant(ty, 1), f"{name}_neg")
            return notv
        return self._iadd3_coerce_int(IRBuilder, _get_val(op, name), ty, name)

    def _iadd3_coerce_int(self, IRBuilder, v, ty, name):
        """Coerce `v` to integer type `ty` (bitcast floats, then int-cast width)."""
        if str(v.type) == 'float':
            v = IRBuilder.bitcast(v, self.ir.IntType(32), f"{name}_bitcast")
        elif str(v.type) == 'double':
            v = IRBuilder.bitcast(v, self.ir.IntType(64), f"{name}_bitcast")
        return self._insert_type_cast_if_needed(IRBuilder, v, ty, f"{name}_narrow")

    def _ensure_int_for_zext(self, IRBuilder, val, i64, name):
        """Ensure val is an integer type suitable for zext to i64."""
        if str(val.type) == 'float':
            val = IRBuilder.bitcast(val, self.ir.IntType(32), f"{name}_bitcast")
        elif str(val.type) == 'double':
            val = IRBuilder.bitcast(val, self.ir.IntType(64), f"{name}_bitcast")
            return val  # already i64
        elif not hasattr(val.type, 'width'):
            val = self._insert_type_cast_if_needed(IRBuilder, val, self.ir.IntType(32), f"{name}_toi32")
        return IRBuilder.zext(val, i64, f"{name}_64")

    @lift_for("IADD3", "UIADD3", "IADD364")
    def lift_iadd3(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        # max carry out from three additions need two bits
        # carry out = cout1 + cout2
        cout1 = Inst.get_defs()[1] if len(Inst.get_defs()) > 1 else None
        cout2 = Inst.get_defs()[2] if len(Inst.get_defs()) > 2 else None
        uses = Inst.get_uses()
        cin1 = uses[3] if len(uses) > 3 else None
        cin2 = uses[4] if len(uses) > 4 else None

        # Fast path: plain 32-bit IADD3 with no carry-OUT def. The widely-used
        # i64 round-trip (zext all operands -> add i64 -> trunc) is only needed
        # to expose bit 32 for the carry-out; without it, a native i32 add is
        # bit-identical and avoids the cvt.u64.u32/add.s64/cvt.u32.u64 inflation
        # (~2-3x register pressure) that tanks occupancy on hot atomic/mem loops
        # (filter, atomicPerf, threadfence, ... timeout cluster).
        dest_ty = dest.get_ir_type(self)
        if cout1 is None and getattr(dest_ty, "width", None) == 32:
            v1 = self._iadd3_operand_narrow(IRBuilder, _get_val, uses[0], dest_ty, "iadd3_o1")
            v2 = self._iadd3_operand_narrow(IRBuilder, _get_val, uses[1], dest_ty, "iadd3_o2")
            v3 = self._iadd3_operand_narrow(IRBuilder, _get_val, uses[2], dest_ty, "iadd3_o3")
            acc = IRBuilder.add(v1, v2, "iadd3_n1")
            acc = IRBuilder.add(acc, v3, "iadd3_n2")
            for cin, cname in ((cin1, "iadd3_cin1"), (cin2, "iadd3_cin2")):
                if cin is not None:
                    cv = self._iadd3_coerce_int(IRBuilder, _get_val(cin, cname), dest_ty, cname)
                    acc = IRBuilder.add(acc, cv, f"{cname}_add")
            self._set_val(IRBuilder, IRRegs, dest, acc)
            return

        v1_64 = self._iadd3_get_operand(IRBuilder, _get_val, uses[0], "iadd3_o1")
        v2_64 = self._iadd3_get_operand(IRBuilder, _get_val, uses[1], "iadd3_o2")
        v3_64 = self._iadd3_get_operand(IRBuilder, _get_val, uses[2], "iadd3_o3")

        tmp = IRBuilder.add(v1_64, v2_64, "iadd3_tmp")
        tmp = IRBuilder.add(tmp, v3_64, "iadd3_tmp2")

        i64 = self.ir.IntType(64)
        if cin1 is not None:
            cin1_val = _get_val(cin1, "iadd3_cin1")
            cin1_64 = self._ensure_int_for_zext(IRBuilder, cin1_val, i64, "iadd3_cin1")
            tmp = IRBuilder.add(tmp, cin1_64, "iadd3_tmp3")

        if cin2 is not None:
            cin2_val = _get_val(cin2, "iadd3_cin2")
            cin2_64 = self._ensure_int_for_zext(IRBuilder, cin2_val, i64, "iadd3_cin2")    
            tmp = IRBuilder.add(tmp, cin2_64, "iadd3_tmp4")        
        
        tmp_trunc = IRBuilder.trunc(tmp, dest.get_ir_type(self), "iadd3_trunc")
        self._set_val(IRBuilder, IRRegs, dest, tmp_trunc)
        
        # carry out (if there is predicate defs)
        if cout1 is not None:
            carry_val_64 = IRBuilder.lshr(tmp, self.ir.Constant(self.ir.IntType(64), dest.get_ir_type(self).width), "iadd3_carry_val")
            # cout1 is true if carry >= 1
            one_wc = self.ir.Constant(self.ir.IntType(64), 1)
            c1_cond = IRBuilder.icmp_unsigned(">=", carry_val_64, one_wc, "carry_ge_1")
            self._set_val(IRBuilder, IRRegs, cout1, c1_cond)
            if cout2 is not None:
                # cout2 is true if carry >= 2
                two_wc = self.ir.Constant(self.ir.IntType(64), 2)
                c2_cond = IRBuilder.icmp_unsigned(">=", carry_val_64, two_wc, "carry_ge_2")
                self._set_val(IRBuilder, IRRegs, cout2, c2_cond)

    @lift_for("ISUB")
    def lift_isub(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "isub_lhs")
        v2 = _get_val(uses[1], "isub_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "sub"))

    @lift_for("SHL", "USHL")
    def lift_shl(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "shl_lhs"), i32, "shl_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "shl_rhs"), i32, "shl_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.shl(v1, v2, "shl"))

    @lift_for("SHL64")
    def lift_shl64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shl_lhs_64")
        v2 = _get_val(uses[1], "shl_rhs_64")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.shl(v1, v2, "shl"))

    @lift_for("SHR", "SHR64", "USHR")
    def lift_shr(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "shr_lhs"), i32, "shr_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "shr_rhs"), i32, "shr_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.lshr(v1, v2, "shr"))

    @lift_for("SHF", "USHF")
    def lift_shf(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "shf_lo")
        v2 = _get_val(uses[1], "shf_shift")
        v3 = _get_val(uses[2], "shf_hi")

        lo64 = IRBuilder.zext(v1, self.ir.IntType(64))
        hi64 = IRBuilder.zext(v3, self.ir.IntType(64))
        v = IRBuilder.or_(
            lo64,
            IRBuilder.shl(hi64, self.ir.Constant(self.ir.IntType(64), 32)),
            "shf_concat",
        )

        v2 = IRBuilder.zext(v2, self.ir.IntType(64), "shf_shift_64")

        mods = Inst.opcodes[1:]
        right = Inst.opcodes[1] == "R"
        # Type/sign token is whichever modifier carries S/U (not "W"/"HI"/"L"),
        # so .W (wrap) shifting the token order doesn't misread the sign.
        signed = any(t.startswith("S") for t in mods)
        is64 = any("64" in t for t in mods)
        high = "HI" in mods
        # `.W` (wrap): the HW masks the shift count to the element width
        # (mod 32 for b32, mod 64 for b64). nvcc folds a source `& 0x1f` into
        # this, passing the UNMASKED value as the count (SHF.R.W.U32.HI RZ,h,x
        # for `x >> (h & 0x1f)`). Without masking, lifting shifts by the full h
        # -> 0/garbage (bloom-filter bit tests in compute-score etc.). The
        # non-.W (clamp) form caps the count at the width, which a 64-bit lshr
        # already approximates (count>64 -> 0).
        if "W" in mods:
            wrap_mask = (64 if is64 else 32) - 1
            v2 = IRBuilder.and_(v2, self.ir.Constant(self.ir.IntType(64), wrap_mask),
                                "shf_wrap")
        if right:
            if signed:
                r = IRBuilder.ashr(v, v2)
            else:
                r = IRBuilder.lshr(v, v2)
        else:
            r = IRBuilder.shl(v, v2)

        if high:
            r = IRBuilder.lshr(r, self.ir.Constant(self.ir.IntType(64), 32), "shf_hi")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.trunc(
            r, dest.get_ir_type(self), "shf_result"
        ))

    @lift_for("IADD")
    def lift_iadd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iadd_lhs")
        v2 = _get_val(uses[1], "iadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd"))

    @lift_for("BMSK")
    def lift_bmsk(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        start = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "bmsk_start"), i32, "bmsk_start")
        width = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "bmsk_width"), i32, "bmsk_width")
        one = self.ir.Constant(i32, 1)
        mask = IRBuilder.shl(one, width, "bmsk_shl")
        mask = IRBuilder.sub(mask, one, "bmsk_sub")
        val = IRBuilder.shl(mask, start, "bmsk")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("VIMNMX3")
    def lift_vimnmx3(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        a = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0], "vimnmx3_a"), i32, "vimnmx3_a")
        b = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1], "vimnmx3_b"), i32, "vimnmx3_b")
        c = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[2], "vimnmx3_c"), i32, "vimnmx3_c")
        pred = uses[3] if len(uses) > 3 else None
        # VIMNMX3: PT → min(a,b,c), !PT → max — SAME polarity as IMNMX/VIMNMX/
        # VIADDMNMX (PT=min, !PT=max; IMNMX convention verified empirically,
        # BUG-090). The old code had a spurious leading `not`, inverting this
        # (PT→max/!PT→min): pns's byte-max `max=max<temp?temp:max` compiles to
        # VIMNMX3 ...!PT, which the inverted form lowered to MIN, collapsing the
        # max to 0 (mean_maxs=0 while the sibling sum mean_vars stayed nonzero).
        lane_bits = self._vsimd_lane_bits(Inst.opcodes)
        is_unsigned = self._vsimd_unsigned(Inst.opcodes)
        is_min = bool(pred and pred.is_pt and not pred.is_not_reg)
        m1 = self._vsimd_minmax(IRBuilder, a, b, lane_bits, is_unsigned, is_min, "vimnmx3_m1")
        val = self._vsimd_minmax(IRBuilder, m1, c, lane_bits, is_unsigned, is_min, "vimnmx3")
        val = self._vsimd_relu(IRBuilder, val, Inst.opcodes, "vimnmx3")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("VIADD")
    def lift_viadd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        v1 = _get_val(uses[0], "viadd_lhs")
        v2 = _get_val(uses[1], "viadd_rhs")
        v1 = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "viadd_lhs")
        v2 = self._insert_type_cast_if_needed(IRBuilder, v2, i32, "viadd_rhs")
        # VIADD.16x2 is a packed per-lane add (no inter-lane carry).
        val = self._vsimd_add(IRBuilder, v1, v2, self._vsimd_lane_bits(Inst.opcodes), "viadd")
        self._set_val(IRBuilder, IRRegs, dest, val)

    def _sel_operand_val(self, Inst, op, IRBuilder, _get_val, tag):
        """Materialize an FSEL/SEL source operand.

        FSEL/SEL is a 32-bit *lane* select: it moves one of two 32-bit register
        lanes into a 32-bit destination, with no notion of f32 vs f64. The
        disassembler renders a NaN-bit immediate as the symbolic token `+QNAN`,
        which the operand parser collapses to a typed float NaN — and that
        always lowers to the 32-bit-float QNAN pattern 0x7fc00000. That is wrong
        when the lane is the *high word* of an f64 (cuBLAS's f64 NaN-guard
        idiom: `FSEL Rhi, RZ, +QNAN, P`), where the real encoded immediate is
        0x7ff80000 (the high 32 bits of a double QNAN). Using 0x7fc00000 there
        yields {hi=0x7fc00000, lo=0} = a *finite* 2.2e307, not a NaN, which then
        flips the downstream `DSETP.GEU` unordered compare and corrupts the
        result (cuBLAS DGEMV: all outputs become 2.2e307 instead of A·x).

        The encoded instruction word holds the true 32-bit immediate in bits
        [63:32] (f32 FSEL: 0x7fc00000; f64 FSEL: 0x7ff80000), so recover it from
        Inst.encoding rather than the lossy `+QNAN` token. For f32 this is a
        no-op (same 0x7fc00000), so the passing f32 path is unaffected.
        """
        if op.is_immediate and op.immediate_value is not None:
            iv = op.immediate_value
            # NaN is the only value not equal to itself; avoids a math import.
            if isinstance(iv, float) and iv != iv and Inst.encoding:
                word = int(Inst.encoding, 16)
                imm32 = (word >> 32) & 0xFFFFFFFF
                i32_const = self.ir.Constant(self.ir.IntType(32), imm32)
                return IRBuilder.bitcast(i32_const, self.ir.FloatType(), tag)
        return _get_val(op, tag)

    @lift_for("SEL", "FSEL", "USEL", "UFSEL")
    def lift_sel(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = self._sel_operand_val(Inst, uses[0], IRBuilder, _get_val, "sel_true")
        v2 = self._sel_operand_val(Inst, uses[1], IRBuilder, _get_val, "sel_false")
        pred = _get_val(uses[2], "sel_pred")
        # pred is a predicate (i1) in SASS, but _get_val may have widened it
        # to i32 when the source pseudo-register was typed as int. select
        # requires an i1 (or vector-of-i1) condition.
        pred = self._normalize_i1_in_curr_bb(IRBuilder, pred, "sel_pred")
        # select requires matching operand types; if only one arm came back as
        # the recovered f32 immediate, coerce the other to match its bit width.
        if v1.type != v2.type:
            v2 = self._insert_type_cast_if_needed(IRBuilder, v2, v1.type, "sel_false")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.select(pred, v1, v2, "sel"))

    @lift_for("IADD64")
    def lift_iadd64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "iadd_lhs")
        v2 = _get_val(uses[1], "iadd_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd"))

    @lift_for("IADD32I", "IADD32I64")
    def lift_iadd32i(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        op1, op2 = uses[0], uses[1]
        v1 = _get_val(op1, "iadd32i_lhs")

        def sx(v, n):
            v &= (1 << n) - 1
            return (v ^ (1 << (n - 1))) - (1 << (n - 1))

        # IADD32I's immediate is a signed 24-bit field; Operand has no setter —
        # write the sign-extended value into immediate_value directly.
        op2.immediate_value = sx(int(op2.name, 16), 24)
        v2 = _get_val(op2, "iadd32i_rhs")
        self._set_val(IRBuilder, IRRegs, dest, IRBuilder.add(v1, v2, "iadd32i"))

    @lift_for("LOP", "ULOP")
    def lift_lop(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        subop = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None
        vb = _get_val(b, "lop_b")

        if subop == "PASS_B":
            self._set_val(IRBuilder, IRRegs, dest, vb)
        else:
            raise UnsupportedInstructionException

    @lift_for("LOP32I", "ULOP32I")
    def lift_lop32i(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        v1 = _get_val(a, "lop32i_a")
        v2 = _get_val(b, "lop32i_b")
        func = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None

        if func == "AND":
            self._set_val(IRBuilder, IRRegs, dest, IRBuilder.and_(v1, v2, "lop32i_and"))
        else:
            raise UnsupportedInstructionException

    @lift_for("BFE")
    def lift_bfe(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("BFI")
    def lift_bfi(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("SSY", "BSSY")
    def lift_ssy(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("SYNC", "BSYNC")
    def lift_sync(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("WARPSYNC")
    def lift_warpsync(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("NOP", "YIELD", "ERRBAR", "LDGDEPBAR", "ENDCOLLECTIVE", "CGAERRBAR",
              "UTMACCTL", "UTMACMDFLUSH", "UTMAPF", "UCGABAR_ARV", "UCGABAR_WAIT",
              "CCTL")
    def lift_nop(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("BMOV")
    def lift_bmov(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # BMOV moves between barrier registers (Bn) and general registers,
        # or clears a barrier and stores its value:
        #   BMOV.32.CLEAR Rd, Bn   - clear Bn, write old value to Rd
        #   BMOV.32       Bn, Rs   - copy Rs to Bn (barrier)
        #   BMOV.16       ...      - 16-bit variants
        # Barrier registers are opaque convergence-tracking state with no
        # observable single-thread semantics, so we don't model them. But
        # the GR-side destination is real: any def of a writable register
        # must populate ir_regs or downstream uses (STL, etc.) hit a
        # KeyError. Zero-initialize any GR defs.
        for d in Inst.get_defs():
            if d is None or not d.is_writable_reg:
                continue
            zero = self.ir.Constant(d.get_ir_type(self), 0)
            self._set_val(IRBuilder, IRRegs, d, zero)

    @lift_for("BPT")
    def lift_bpt(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        self.run_intrinsic("trap", IRBuilder, [], "trap")

    @lift_for("BRA")
    def lift_bra(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        uses = Inst.get_uses()
        target_addr = uses[-1].immediate_value

        # Check for conditional BRA with predicate operand (e.g. BRA P1, target
        # from @!P0 BRA P1, target after pflag stripping in predicated blocks)
        has_pred_operand = (
            len(uses) >= 2
            and uses[0].is_predicate_reg
        )

        target_bb = None
        if Inst.parent and getattr(Inst.parent, "succs", None):
            for succ in Inst.parent.succs:
                if succ.addr == target_addr:
                    target_bb = succ
                    break

        if target_bb is None:
            for bb in block_map.keys():
                if bb.addr == target_addr:
                    target_bb = bb
                    break

        if target_bb is None:
            raise Exception(f"Cannot resolve BRA target: {uses[-1]}")

        if has_pred_operand:
            # Conditional branch: BRA Pred, target
            cond = _get_val(uses[0], "bra_cond")
            # cond is a predicate (i1) in SASS, but _get_val may have widened
            # it to i32 when the source pseudo-register was typed as int.
            # cbranch requires i1.
            cond = self._normalize_i1_in_curr_bb(IRBuilder, cond, "bra_cond")
            # Find fall-through block (the other successor)
            fall_bb = None
            if Inst.parent and getattr(Inst.parent, "succs", None):
                for succ in Inst.parent.succs:
                    if succ is not target_bb:
                        fall_bb = succ
                        break
            if fall_bb is None:
                # No fallthrough — treat as unconditional
                IRBuilder.branch(block_map[target_bb])
            else:
                IRBuilder.cbranch(cond, block_map[target_bb], block_map[fall_bb])
        else:
            IRBuilder.branch(block_map[target_bb])
        

    @lift_for("SWITCH")
    def lift_switch(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        sel_op = Inst.get_uses()[0]
        sel_val = _get_val(sel_op, "brx_sel")

        # Use the ordered target list stored by ResolveBRX.  This preserves
        # the original case→target mapping including duplicates that the
        # deduplicated succs list loses.
        targets = getattr(Inst, "switch_targets", None) or Inst.parent.succs
        # Last entry is the out-of-range / default target.
        default_bb = targets[-1]
        sw = IRBuilder.switch(sel_val, block_map[default_bb])
        for idx, tgt in enumerate(targets[:-1]):
            # Skip cases that go to the default block — the default edge
            # already covers them.  This avoids duplicate LLVM IR edges
            # which would require duplicate PHI entries.
            if tgt is default_bb:
                continue
            sw.add_case(self.ir.Constant(sel_val.type, idx), block_map[tgt])

    @lift_for("BRK")
    def lift_brk(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("IMNMX")
    def lift_imnmx(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # SASS form: IMNMX Rd, Rs1, Rs2, P
        #   P == PT  →  Rd = min(Rs1, Rs2)
        #   P == !PT →  Rd = max(Rs1, Rs2)
        # Convention verified empirically by compiling
        #   *out_min = (*a < *b) ? *a : *b;   →  IMNMX ..., PT
        #   *out_max = (*a > *b) ? *a : *b;   →  IMNMX ..., !PT
        # on sm_80 (cuBLAS 13.1.0.3). Earlier two attempts had this
        # backwards (BUG-090): the cuBLAS robust-hypot reduction uses
        # `R10 = MAX(R2,R5), R12 = MIN(R2,R5)` then gates the final FMUL
        # via FSETP.NEU(R12, 0). With the convention flipped, R10/R12
        # were swapped, so the gate never caught the (small=0, max≠0)
        # case and the merge overflowed.
        #
        # Note: pred.is_pt only checks the register name ("PT"); it does
        # NOT distinguish PT from !PT. Both share is_pt=True, so the
        # negation prefix must be checked separately via is_not_reg.
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0], "imnmx_lhs")
        v2 = _get_val(uses[1], "imnmx_rhs")
        pred = uses[2]

        isUnsigned = "U32" in Inst.opcodes
        isMax = bool(pred.is_not_reg)   # !PT = max, PT (no neg) = min

        if isUnsigned:
            if isMax:
                cond = IRBuilder.icmp_unsigned(">", v1, v2, "imnmx_cmp")
            else:
                cond = IRBuilder.icmp_unsigned("<", v1, v2, "imnmx_cmp")
        else:
            if isMax:
                cond = IRBuilder.icmp_signed(">", v1, v2, "imnmx_cmp")
            else:
                cond = IRBuilder.icmp_signed("<", v1, v2, "imnmx_cmp")

        self._set_val(IRBuilder, IRRegs, dest,
                      IRBuilder.select(cond, v1, v2, "imnmx_max"))

    @lift_for("PSETP", "UPSETP64")
    def lift_psetp(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("PBK")
    def lift_pbk(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        raise UnsupportedInstructionException

    @lift_for("LEA", "LEA64", "ULEA")
    def lift_lea(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        dest = defs[0]

        is_hi = "HI" in Inst.opcodes
        is_sx32 = "SX32" in Inst.opcodes
        has_cin = "X" in Inst.opcodes
        has_cout = len(defs) > 1

        val_src = _get_val(uses[0], "lea_src")
        val_base = _get_val(uses[1], "lea_base")

        if is_sx32:
            val_shift = _get_val(uses[2], "lea_shift")
            val_a_64 = IRBuilder.sext(val_src, self.ir.IntType(64))
        elif is_hi:
            # LEA.HI Rd, Ra, Rb, Rc, shift[, carry] is a FUNNEL shift:
            #   Rd = (((zext(Rc) << 32 | zext(Ra)) << shift) >> 32) + Rb + carry
            # The funnel's HIGH word is Rc (uses[2]); dropping it loses the
            # (Ra << shift) contribution of a rotate idiom (e.g. chacha20's
            # quarter-round rotl, emitted as LEA.HI with Rc==Ra) -> wrong result.
            # (LEA.HI.X.SX32 address math has no separate Rc and is handled by
            # the is_sx32 branch above, checked first.)
            val_shift = _get_val(uses[3], "lea_shift")
            lo64 = IRBuilder.zext(val_src, self.ir.IntType(64))
            hi64 = IRBuilder.zext(_get_val(uses[2], "lea_hi"), self.ir.IntType(64))
            val_a_64 = IRBuilder.or_(
                lo64,
                IRBuilder.shl(hi64, self.ir.Constant(self.ir.IntType(64), 32)),
                "lea_funnel",
            )
        else:
            val_shift = _get_val(uses[2], "lea_shift")
            val_a_64 = IRBuilder.zext(val_src, self.ir.IntType(64))

        val_base_64 = IRBuilder.zext(val_base, self.ir.IntType(64))
        val_shift_64 = IRBuilder.zext(val_shift, self.ir.IntType(64))

        val_shifted = IRBuilder.shl(val_a_64, val_shift_64)

        if is_hi:
            val_shifted_high = IRBuilder.lshr(val_shifted, self.ir.Constant(self.ir.IntType(64), 32))
            t_sum = IRBuilder.add(val_shifted_high, val_base_64)

            if has_cin:
                val_cin = _get_val(uses[-1], "lea_cin")
                t_sum = IRBuilder.add(t_sum, IRBuilder.zext(val_cin, self.ir.IntType(64)))

            res = IRBuilder.trunc(t_sum, dest.get_ir_type(self))
        else:
            val_shifted_low = IRBuilder.and_(val_shifted, self.ir.Constant(self.ir.IntType(64), 0xFFFFFFFF))
            t_sum = IRBuilder.add(val_shifted_low, val_base_64)

            res = IRBuilder.trunc(t_sum, dest.get_ir_type(self))

        if has_cout:
            carry_val = IRBuilder.lshr(t_sum, self.ir.Constant(self.ir.IntType(64), 32))
            self._set_val(IRBuilder, IRRegs, defs[1], IRBuilder.trunc(carry_val, self.ir.IntType(1)))

        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("F2I", "UF2I")
    def lift_f2i(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]

        modifiers = set(Inst.opcodes[1:])
        is_unsigned = any(m.startswith("U") and m[1:].isdigit() for m in modifiers)

        # Determine source float type and dest int width from opcodes
        # F2I.F16.NTZ       → src=F16, dest=S32 implicit
        # F2I.U32.F16.NTZ   → dest=U32, src=F16
        # F2I.S16.F16.NTZ   → dest=S16, src=F16
        # F2I.FLOOR.NTZ     → src=F32 implicit, dest=S32 implicit
        float_type_map = {"F16": self.ir.HalfType(), "F32": self.ir.FloatType(), "F64": self.ir.DoubleType()}
        src_float = None
        dest_width = 32  # default
        for m in Inst.opcodes[1:]:
            if m in float_type_map:
                src_float = float_type_map[m]
            elif m.startswith("S") and m[1:].isdigit():
                dest_width = int(m[1:])
            elif m.startswith("U") and m[1:].isdigit():
                dest_width = int(m[1:])

        i32 = self.ir.IntType(32)
        v1 = _get_val(op1, "f2i_src")

        # If source is F16, extract lower 16 bits and bitcast to half
        if isinstance(src_float, self.ir.HalfType):
            v1_i32 = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "f2i_src_i32")
            v1_i16 = IRBuilder.trunc(v1_i32, self.ir.IntType(16), "f2i_src_i16")
            v1 = IRBuilder.bitcast(v1_i16, self.ir.HalfType(), "f2i_src_half")
        elif not isinstance(v1.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
            v1 = IRBuilder.bitcast(v1, self.ir.FloatType(), "f2i_src_cast")

        # fptosi/fptoui truncate toward zero, but F2I's default rounding is
        # round-to-nearest-even. Round the float per the modifier before the
        # (truncating) convert.
        if not ("TRUNC" in modifiers or "RZ" in modifiers):
            if "FLOOR" in modifiers or "RM" in modifiers:
                rfn = "floor"
            elif "CEIL" in modifiers or "RP" in modifiers:
                rfn = "ceil"
            else:
                rfn = "rint"  # round-to-nearest-even (F2I default)
            if isinstance(v1.type, self.ir.HalfType):
                v1 = IRBuilder.fpext(v1, self.ir.FloatType(), "f2i_h2f"); _suf = "f32"
            elif isinstance(v1.type, self.ir.DoubleType):
                _suf = "f64"
            else:
                _suf = "f32"
            v1 = self.run_intrinsic(f"{rfn}_{_suf}", IRBuilder, [v1], f"f2i_{rfn}")

        dest_int_ty = self.ir.IntType(dest_width)
        if is_unsigned:
            val = IRBuilder.fptoui(v1, dest_int_ty, "f2i")
        else:
            val = IRBuilder.fptosi(v1, dest_int_ty, "f2i")

        # Zero/sign-extend to i32 if dest_width < 32
        if dest_width < 32:
            if is_unsigned:
                val = IRBuilder.zext(val, i32, "f2i_zext")
            else:
                val = IRBuilder.sext(val, i32, "f2i_sext")

        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("F2IP")
    def lift_f2ip(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # F2IP = float-to-int pack-pair: ptxas's fused lowering of the PTX
        # `cvt.pack.sat.s8.s32.b32 d, a, b, c` when the s32 inputs are themselves
        # `cvt.rni.s32.f32` of floats (the CUTLASS NumericArrayConverter<int8_t,
        # float> epilogue that quantizes fp32 accumulators to packed int8). The
        # int8 GEMM examples (13_fused_two_gemms_s8) emit it after the IMMA mainloop.
        #
        #   F2IP.S8.F32.NTZ Rd, Ra, Rb, Rc   →
        #     Rd[ 7: 0] = sat_s8( rne(Rb) )    (SECOND source → byte0)
        #     Rd[15: 8] = sat_s8( rne(Ra) )    (FIRST  source → byte1)
        #     Rd[31:16] = Rc[15:0]             (accumulator low half → bytes 2,3)
        #
        # Byte order verified against the PTX ISA + the CUTLASS 4-wide converter
        # (cvt.pack r4,%4,%3,0; cvt.pack out,%2,%1,r4) and the microbench dataflow:
        # two F2IP per 32-bit reg fill all four bytes (the 2nd takes Rc=1st's Rd, so
        # the 1st pair slides from byte{0,1} into byte{2,3}).
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()

        i32 = self.ir.IntType(32)
        i16 = self.ir.IntType(16)

        # Destination integer width and signedness from the .<S8|U8|S16|U16> token.
        # The corpus form is .S8 (signed int8). Default S8 if absent.
        dst_width, dst_signed = 8, True
        for m in Inst.opcodes[1:]:
            if m and m[0] in ("S", "U") and m[1:].isdigit():
                dst_width = int(m[1:])
                dst_signed = (m[0] == "S")
                break

        # Source float type from the .<F16|F32|F64> token (corpus = .F32).
        float_type_map = {"F16": self.ir.HalfType(),
                          "F32": self.ir.FloatType(),
                          "F64": self.ir.DoubleType()}
        src_float = self.ir.FloatType()
        for m in Inst.opcodes[1:]:
            if m in float_type_map:
                src_float = float_type_map[m]
                break

        # Saturation bounds for the packed sub-word.
        if dst_signed:
            lo_bound, hi_bound = -(1 << (dst_width - 1)), (1 << (dst_width - 1)) - 1
        else:
            lo_bound, hi_bound = 0, (1 << dst_width) - 1

        def convert_one(use_op, nm):
            """rne-round + saturate one float source → an i32 holding the
            (sign/zero-extended) dst_width-bit converted value in its low byte(s)."""
            v = _get_val(use_op, f"f2ip_{nm}_src")
            # Bring the operand to its true float type.
            if isinstance(src_float, self.ir.HalfType):
                v_i32 = self._insert_type_cast_if_needed(IRBuilder, v, i32, f"f2ip_{nm}_i32")
                v_i16 = IRBuilder.trunc(v_i32, i16, f"f2ip_{nm}_i16")
                v = IRBuilder.bitcast(v_i16, self.ir.HalfType(), f"f2ip_{nm}_half")
            elif not isinstance(v.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                v = IRBuilder.bitcast(v, src_float, f"f2ip_{nm}_cast")
            # F2IP default rounding is round-to-nearest-even (.NTZ). fptosi/fptoui
            # truncate, so round explicitly first (same policy as lift_f2i).
            rsuf = "f64" if isinstance(v.type, self.ir.DoubleType) else "f32"
            if isinstance(v.type, self.ir.HalfType):
                v = IRBuilder.fpext(v, self.ir.FloatType(), f"f2ip_{nm}_h2f")
            v = self.run_intrinsic(f"rint_{rsuf}", IRBuilder, [v], f"f2ip_{nm}_rint")
            # Convert to a wide i32 (no overflow for sane magnitudes), then clamp to
            # the dst_width saturation range — .S8/.U8 saturate (satfinite).
            if dst_signed:
                iv = IRBuilder.fptosi(v, i32, f"f2ip_{nm}_i")
            else:
                iv = IRBuilder.fptoui(v, i32, f"f2ip_{nm}_i")
            hi_c = self.ir.Constant(i32, hi_bound)
            lo_c = self.ir.Constant(i32, lo_bound)
            gt = IRBuilder.icmp_signed(">", iv, hi_c, f"f2ip_{nm}_gt")
            iv = IRBuilder.select(gt, hi_c, iv, f"f2ip_{nm}_clamphi")
            lt = IRBuilder.icmp_signed("<", iv, lo_c, f"f2ip_{nm}_lt")
            iv = IRBuilder.select(lt, lo_c, iv, f"f2ip_{nm}_clamplo")
            # Keep only the low dst_width bits (the byte that gets packed).
            mask = self.ir.Constant(i32, (1 << dst_width) - 1)
            return IRBuilder.and_(iv, mask, f"f2ip_{nm}_byte")

        # Ra → byte1 (bits [15:8]); Rb → byte0 (bits [7:0]).
        a_byte = convert_one(uses[0], "a")
        b_byte = convert_one(uses[1], "b")
        a_shifted = IRBuilder.shl(a_byte, self.ir.Constant(i32, dst_width), "f2ip_a_shift")
        packed_lo = IRBuilder.or_(a_shifted, b_byte, "f2ip_pack_lo")

        # Rc[15:0] → the high 16 bits of the result (bytes 2,3). RZ → 0.
        if len(uses) >= 3:
            c_val = _get_val(uses[2], "f2ip_acc")
            c_val = self._insert_type_cast_if_needed(IRBuilder, c_val, i32, "f2ip_acc_i32")
            c_lo16 = IRBuilder.and_(c_val, self.ir.Constant(i32, 0xFFFF), "f2ip_acc_lo16")
            c_hi = IRBuilder.shl(c_lo16, self.ir.Constant(i32, 16), "f2ip_acc_hi")
            result = IRBuilder.or_(c_hi, packed_lo, "f2ip_packed")
        else:
            result = packed_lo

        self._set_val(IRBuilder, IRRegs, dest, result)

    def _int_to_float_directed(self, IRBuilder, ival, float_ty, is_unsigned, rnd):
        """int -> float with directed rounding (rnd = "RP" toward +Inf, "RM"
        toward -Inf).  Returns a float-typed value.

        LLVM's sitofp/uitofp are round-to-nearest-even; for |int| beyond the
        float's exact range (>=2^24 for f32, >=2^53 for f64) the rounding
        direction is observable.  Algorithm: convert RN, convert back to the
        int, and if RN landed on the wrong side of the true value nudge by one
        ULP in the requested direction (nextafter).  Integers have no fractional
        part, so the directed result is the nearest float on that side; the only
        correction needed is a single ULP step when RN rounded the other way.

        The round-trip (`fptosi`/`fptoui` of the RN result) is done in a
        DOUBLE-WIDTH int: rounding up can push the RN result past the source
        int's max (e.g. i32 max 2^31-1 rounds to 2^31), and converting that back
        into the same width would overflow to poison and corrupt the comparison.
        """
        f_rn = (IRBuilder.uitofp(ival, float_ty, "i2f_rn")
                if is_unsigned else IRBuilder.sitofp(ival, float_ty, "i2f_rn"))
        # Compare in a 2x-wide int so a rounded-up f_rn never overflows on the
        # way back.  Extend `ival` to the same wide type for the comparison.
        wide_ty = self.ir.IntType(ival.type.width * 2)
        ival_w = (IRBuilder.zext(ival, wide_ty, "i2f_iw")
                  if is_unsigned else IRBuilder.sext(ival, wide_ty, "i2f_iw"))
        back = (IRBuilder.fptoui(f_rn, wide_ty, "i2f_back")
                if is_unsigned else IRBuilder.fptosi(f_rn, wide_ty, "i2f_back"))
        ival = ival_w
        cmp = IRBuilder.icmp_unsigned if is_unsigned else IRBuilder.icmp_signed
        fbits_ty = self.ir.IntType(32 if isinstance(float_ty, self.ir.FloatType)
                                   else (16 if isinstance(float_ty, self.ir.HalfType)
                                         else 64))
        fbits = IRBuilder.bitcast(f_rn, fbits_ty, "i2f_fbits")
        one = self.ir.Constant(fbits_ty, 1)
        sign_neg = IRBuilder.icmp_signed("<", fbits, self.ir.Constant(fbits_ty, 0),
                                         "i2f_fneg")
        if rnd == "RP":
            # round toward +Inf: needed only when RN landed BELOW the true value.
            need = cmp("<", back, ival, "i2f_rp_lt")
            # nextafter(+Inf): magnitude up.  For f>=0 bits+1; for f<0 bits-1.
            up = IRBuilder.select(sign_neg,
                                  IRBuilder.sub(fbits, one, "i2f_up_neg"),
                                  IRBuilder.add(fbits, one, "i2f_up_pos"),
                                  "i2f_nextup")
        else:  # "RM" — round toward -Inf: needed only when RN landed ABOVE.
            need = cmp(">", back, ival, "i2f_rm_gt")
            # nextafter(-Inf): magnitude down.  For f>=0 bits-1; for f<0 bits+1.
            up = IRBuilder.select(sign_neg,
                                  IRBuilder.add(fbits, one, "i2f_dn_neg"),
                                  IRBuilder.sub(fbits, one, "i2f_dn_pos"),
                                  "i2f_nextdn")
        corrected = IRBuilder.bitcast(up, float_ty, "i2f_corr")
        return IRBuilder.select(need, corrected, f_rn, "i2f_directed")

    @lift_for("I2F", "I2FP", "UI2F", "UI2FP")
    def lift_i2f(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]

        # Scan the modifier tokens by KIND rather than position — a trailing
        # rounding flag (.RP/.RM/.RZ) must not be mistaken for the source type.
        #   I2FP.F32.S32 → dest=F32, src=S32, RN
        #   I2F.S16      → dest=F32 (implicit), src=S16, RN
        #   I2F.F16      → dest=F16, src=S32 (implicit), RN
        #   I2F.RP       → dest=F32, src=S32 (implicit), round +Inf
        #   I2F.U32.RP   → dest=F32, src=U32, round +Inf  (was mis-parsed signed)
        #   I2F.U64.RP   → dest=F32, src=U64, round +Inf  (was mis-parsed S32)
        #   I2F.S64      → dest=F32, src=S64, RN
        float_type_map = {"F16": self.ir.HalfType(), "F32": self.ir.FloatType(),
                          "F64": self.ir.DoubleType()}
        dest_float_type = self.ir.FloatType()   # implicit dest = F32
        src_signed = True                       # implicit src = S32
        src_width = 32
        rnd = None                              # None = RN (default)
        # UI2F/UI2FP opcode prefix forces an unsigned source.
        if Inst.opcodes[0].startswith("UI"):
            src_signed = False
        for m in Inst.opcodes[1:]:
            if m in float_type_map:                         # dest float type
                dest_float_type = float_type_map[m]
            elif m and m[0] in ("S", "U") and m[1:].isdigit():  # src int type
                src_signed = (m[0] == "S")
                src_width = int(m[1:])
            elif m in ("RP", "RM", "RZ", "RN", "RNI"):      # rounding mode
                rnd = m if m in ("RP", "RM") else None
        is_unsigned = not src_signed

        v1 = _get_val(op1, "i2f_src")
        # Source byte/half select modifier: I2F.U8 Rd, Rx.B1 converts BYTE 1 of
        # Rx (not byte 0). The parser keeps it in op.suffix ("B0".."B3"/"H0"/"H1");
        # shift the selected lane into the low bits before the narrow truncate
        # below. (Mirrors the H1 handling in lift_f2fp at ~2343.)
        _bsh = 0
        if op1.suffix == "H1":
            _bsh = 16
        elif op1.suffix and len(op1.suffix) == 2 and op1.suffix[0] == "B" and op1.suffix[1] in "0123":
            _bsh = 8 * int(op1.suffix[1])
        if _bsh:
            i32 = self.ir.IntType(32)
            if v1.type != i32:
                v1 = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "i2f_bsel_w")
            v1 = IRBuilder.lshr(v1, self.ir.Constant(i32, _bsh), "i2f_bsel")
        # Bring the operand to its true int width.  Narrow sources truncate; a
        # 64-bit source widens (PACK64 supplies the i64 value).
        src_int_ty = self.ir.IntType(src_width)
        if v1.type != src_int_ty:
            v1 = self._insert_type_cast_if_needed(IRBuilder, v1, src_int_ty, "i2f_w")

        if rnd in ("RP", "RM"):
            val = self._int_to_float_directed(
                IRBuilder, v1, dest_float_type, is_unsigned, rnd)
        elif is_unsigned:
            val = IRBuilder.uitofp(v1, dest_float_type, "i2f")
        else:
            val = IRBuilder.sitofp(v1, dest_float_type, "i2f")

        # If dest is half, pack into i32 register (zero-extended)
        if isinstance(dest_float_type, self.ir.HalfType):
            i16 = self.ir.IntType(16)
            i32 = self.ir.IntType(32)
            val_i16 = IRBuilder.bitcast(val, i16, "i2f_h_i16")
            val = IRBuilder.zext(val_i16, i32, "i2f_h_i32")

        self._set_val(IRBuilder, IRRegs, dest, val)

    @staticmethod
    def _parse_f2f_types(opcodes):
        """Extract dest and src float types from F2F/F2FP opcode modifiers.

        Handles standard types (F16, F32, F64) and narrow types (BF16, E5M2,
        E4M3).  Returns (dest_type, src_type) strings.  Non-type modifiers
        such as SATFINITE, PACK_AB, PACK_AB_MERGE_C are skipped.
        """
        # Recognised float-type tokens.  Order matters: first match = dest,
        # second = src.
        _FLOAT_TYPES = {"F16", "F32", "F64", "BF16", "E5M2", "E4M3"}
        types_found = [t for t in opcodes[1:] if t in _FLOAT_TYPES]
        if len(types_found) >= 2:
            return types_found[0], types_found[1]
        if len(types_found) == 1:
            # SM80+ shorthand: F2FP.BF16.PACK_AB / F2FP.E5M2.PACK_AB etc.
            # specify only the destination narrow type; source is implicitly
            # F32 (the compute float register width). Same convention as
            # PTX's cvt.rn.bf16.f32.
            return types_found[0], "F32"
        # F2FP.PACK_AB with no explicit type tokens — default to F16 dest,
        # F32 src (the SASS default for F2FP packing on sm_80+).
        if opcodes[0] == "F2FP":
            return "F16", "F32"
        # Fallback — should not happen on well-formed SASS.
        return opcodes[1], opcodes[2]

    @staticmethod
    def _f2f_width(type_token):
        """Map a float type token to its bit-width for lifting purposes.

        BF16 is 16-bit, E5M2/E4M3 (FP8 formats) are 8-bit but are packed
        into 16-bit halfwords during PACK_AB, so we treat them as 8.
        """
        _WIDTHS = {"F16": 16, "F32": 32, "F64": 64, "BF16": 16, "E5M2": 8, "E4M3": 8}
        if type_token in _WIDTHS:
            return _WIDTHS[type_token]
        # Fallback for unknown types: strip first char and parse (e.g. "F16" -> 16)
        return int(type_token[1:])

    def _f32_to_bf16_bits_i32(self, IRBuilder, v, nm, rz=False):
        """f32 -> bf16 at the bit level, result in an i32's low 16 bits.

        bf16 is f32's top half: RN-even = round-add 0x7FFF + lsb then take
        bits [31:16]; RZ = truncate (take bits [31:16] directly).
        fptrunc-to-HALF is NOT bf16 (f16 exp bias 15 vs bf16 127): f16 bits
        read back as bf16 scale by ~2^112 — ex94's bf16 epilogue emitted
        1.45e7 where the value was 14.75. NaN inputs produce the bf16
        canonical NaN 0x7FFF (== CUDART_NAN_BF16): native-H100 validation
        pinned cvt.rn.bf16.f32(0x7FC00000) == 0x7FFF — sign and payload
        are dropped, not quieted-preserved as previously assumed.
        """
        i32 = self.ir.IntType(32)
        u = IRBuilder.bitcast(v, i32, f"{nm}_u32")
        if rz:
            rnd = u
        else:
            lsb = IRBuilder.and_(
                IRBuilder.lshr(u, self.ir.Constant(i32, 16), f"{nm}_l"),
                self.ir.Constant(i32, 1), f"{nm}_lsb")
            rnd = IRBuilder.add(
                IRBuilder.add(u, self.ir.Constant(i32, 0x7FFF),
                              f"{nm}_r1"),
                lsb, f"{nm}_r2")
        bits = IRBuilder.and_(
            IRBuilder.lshr(rnd, self.ir.Constant(i32, 16),
                           f"{nm}_hi"),
            self.ir.Constant(i32, 0xFFFF), f"{nm}_bits")
        is_nan = IRBuilder.fcmp_unordered(
            "uno", v, v, f"{nm}_isnan")
        return IRBuilder.select(is_nan, self.ir.Constant(i32, 0x7FFF),
                                bits, f"{nm}_bf16")

    @lift_for("F2F", "F2FP")
    def lift_f2f(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()

        # F2FP.F16.F32.PACK_AB Rdest, Ra, Rb — convert two F32 values to
        # half and pack into a single i32.
        # F2FP.SATFINITE.E5M2.F32.PACK_AB_MERGE_C Rdest, Ra, Rb, Rc — convert
        # two F32 values to FP8, pack into upper 16 bits, merge with Rc's
        # lower 16 bits.
        # For non-PACK: F2F.F64.F32 Rdest, Rsrc
        is_pack_ab = any(t.startswith("PACK_AB") for t in Inst.opcodes)
        is_merge_c = "PACK_AB_MERGE_C" in Inst.opcodes

        dest_type, src_type = self._parse_f2f_types(Inst.opcodes)
        dest_width = self._f2f_width(dest_type)
        src_width = self._f2f_width(src_type)

        float_type_map = {16: self.ir.HalfType(), 32: self.ir.FloatType(), 64: self.ir.DoubleType()}
        i32 = self.ir.IntType(32)
        i16 = self.ir.IntType(16)

        if is_pack_ab and len(uses) >= 2:
            src_a = _get_val(uses[0], "f2fp_src_a")
            src_b = _get_val(uses[1], "f2fp_src_b")
            src_float_type = float_type_map.get(src_width, self.ir.FloatType())

            if dest_width == 8:
                # FP8 pack: F2FP.{E4M3,E5M2}.F32.PACK_AB[_MERGE_C] Rd, A, B[, C]
                # Encodes A,B to fp8 BYTES (RN-even, satfinite) — NOT the old
                # f16-low-byte hack — and packs them as the PTX fp8x2 layout:
                #   B -> bits [7:0]  (low byte) ;  A -> bits [15:8] (high byte)
                # matching `cvt.e4m3x2.f32 d, a, b` => d[7:0]=cvt(b),d[15:8]=cvt(a).
                if not isinstance(src_a.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                    src_a = IRBuilder.bitcast(src_a, src_float_type, "f2fp_a_cast")
                if not isinstance(src_b.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                    src_b = IRBuilder.bitcast(src_b, src_float_type, "f2fp_b_cast")

                a_i32 = self._f32_to_fp8_byte(IRBuilder, src_a, dest_type, "f2fp_a")
                b_i32 = self._f32_to_fp8_byte(IRBuilder, src_b, dest_type, "f2fp_b")

                a_shifted = IRBuilder.shl(a_i32, self.ir.Constant(i32, 8), "f2fp_a_shift")
                result = IRBuilder.or_(a_shifted, b_i32, "f2fp_packed_lo")

                # MERGE_C: the two new fp8 bytes occupy [15:0]; the OTHER 16-bit
                # half is carried over from the C operand into bits [31:16].
                # `.H1` on C selects C's HIGH 16 bits as that carried half (else
                # the LOW 16).  x86 `_get_val` does not apply the .H1 suffix, so
                # read the RAW C word here (suffix stripped) and select the half
                # explicitly — correct on both x86 and nvvm backends.
                if is_merge_c and len(uses) >= 3:
                    c_op = uses[2]
                    c_suffix = getattr(c_op, "suffix", None)
                    saved = c_op.suffix
                    c_op.suffix = None
                    try:
                        src_c = _get_val(c_op, "f2fp_src_c")
                    finally:
                        c_op.suffix = saved
                    src_c = self._insert_type_cast_if_needed(IRBuilder, src_c, i32, "f2fp_c")
                    if c_suffix == "H1":
                        c_half = IRBuilder.and_(
                            IRBuilder.lshr(src_c, self.ir.Constant(i32, 16), "f2fp_c_h1sh"),
                            self.ir.Constant(i32, 0xFFFF), "f2fp_c_half")
                    else:
                        c_half = IRBuilder.and_(src_c, self.ir.Constant(i32, 0xFFFF), "f2fp_c_half")
                    result = IRBuilder.or_(
                        result,
                        IRBuilder.shl(c_half, self.ir.Constant(i32, 16), "f2fp_c_hi"),
                        "f2fp_merged")

                self._set_val(IRBuilder, IRRegs, dest, result)
                return

            # 16-bit dest (F16, BF16): convert each source, pack two 16-bit
            # patterns into a single i32.  BF16 uses the bit-level RN-even
            # conversion (_f32_to_bf16_bits_i32) — fptrunc-to-half emits f16
            # bits, a ~2^112 scale error when read back as bf16.
            target_half = self.ir.HalfType()

            def to_bf16_bits_i32(v, nm):
                # F2FP.BF16.F32.PACK_AB[.RZ] — ptxas emits the .RZ form for
                # cvt.rz.bf16x2.f32 (and for scalar cvt.rz paired with RZ).
                return self._f32_to_bf16_bits_i32(IRBuilder, v, nm,
                                                  rz="RZ" in Inst.opcodes)

            def to_f16_bits_i32(v, nm):
                h = IRBuilder.fptrunc(v, target_half, f"{nm}_half") \
                    if src_width > 16 else v
                h16 = IRBuilder.bitcast(h, i16, f"{nm}_i16")
                return IRBuilder.zext(h16, i32, f"{nm}_i32")

            is_bf16_dest = dest_type == "BF16"

            # Convert A (high half)
            if not isinstance(src_a.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                src_a = IRBuilder.bitcast(src_a, src_float_type, "f2fp_a_cast")
            h_a_i32 = (to_bf16_bits_i32(src_a, "f2fp_a") if is_bf16_dest
                       else to_f16_bits_i32(src_a, "f2fp_a"))

            # Convert B (low half)
            if not isinstance(src_b.type, (self.ir.HalfType, self.ir.FloatType, self.ir.DoubleType)):
                src_b = IRBuilder.bitcast(src_b, src_float_type, "f2fp_b_cast")
            h_b_i32 = (to_bf16_bits_i32(src_b, "f2fp_b") if is_bf16_dest
                       else to_f16_bits_i32(src_b, "f2fp_b"))

            # Pack: A in upper 16, B in lower 16
            h_a_shifted = IRBuilder.shl(h_a_i32, self.ir.Constant(i32, 16), "f2fp_a_shift")
            result = IRBuilder.or_(h_a_shifted, h_b_i32, "f2fp_packed")
            self._set_val(IRBuilder, IRRegs, dest, result)
            return

        # F2FP.F16.E4M3.UNPACK_B Rd, Rs[.H1] — convert the fp8x2 in the
        # selected 16-bit half of Rs (.H1 = high half, default low) into a
        # packed f16x2. nvdisasm prints only the B source (the A slot is RZ).
        # e4m3 (1s.4e.3m, bias 7, no Inf, S.1111.111=NaN) is an exact subset
        # of f16, so the conversion is pure bit re-encoding: normals remap
        # the exponent (+8), subnormals (m * 2^-9) are f16 normals — computed
        # exactly via uitofp*2^-9->fptrunc. e5m2 is f16's top byte verbatim.
        is_unpack = any(t.startswith("UNPACK") for t in Inst.opcodes)
        if is_unpack and src_width == 8:
            f32 = self.ir.FloatType()
            half = self.ir.HalfType()
            src_op = uses[-1]
            v = _get_val(src_op, "f2fp_unpack_src")
            v = self._insert_type_cast_if_needed(IRBuilder, v, i32, "f2fp_unpack_src")
            if src_op.suffix == "H1":
                v = IRBuilder.lshr(v, self.ir.Constant(i32, 16), "f2fp_unpack_h1")
            halves = []
            for bit, nm in ((0, "lo"), (8, "hi")):
                b = IRBuilder.lshr(v, self.ir.Constant(i32, bit),
                                   f"f2fp_{nm}_sh") if bit else v
                b = IRBuilder.and_(b, self.ir.Constant(i32, 0xFF),
                                   f"f2fp_{nm}_byte")
                if src_type == "E5M2":
                    h = IRBuilder.shl(b, self.ir.Constant(i32, 8),
                                      f"f2fp_{nm}_e5m2")
                else:  # E4M3
                    sgn = IRBuilder.shl(
                        IRBuilder.and_(b, self.ir.Constant(i32, 0x80),
                                       f"f2fp_{nm}_s"),
                        self.ir.Constant(i32, 8), f"f2fp_{nm}_s15")
                    e = IRBuilder.and_(
                        IRBuilder.lshr(b, self.ir.Constant(i32, 3),
                                       f"f2fp_{nm}_esh"),
                        self.ir.Constant(i32, 0xF), f"f2fp_{nm}_e")
                    m = IRBuilder.and_(b, self.ir.Constant(i32, 0x7),
                                       f"f2fp_{nm}_m")
                    norm = IRBuilder.or_(
                        IRBuilder.shl(
                            IRBuilder.add(e, self.ir.Constant(i32, 8),
                                          f"f2fp_{nm}_eb"),
                            self.ir.Constant(i32, 10), f"f2fp_{nm}_ef"),
                        IRBuilder.shl(m, self.ir.Constant(i32, 7),
                                      f"f2fp_{nm}_mf"),
                        f"f2fp_{nm}_norm")
                    # subnormal: m * 2^-9, exact in f16
                    mf = IRBuilder.uitofp(m, f32, f"f2fp_{nm}_mf32")
                    sub_f = IRBuilder.fmul(
                        mf, self.ir.Constant(f32, 2.0 ** -9),
                        f"f2fp_{nm}_subf")
                    sub_h = IRBuilder.fptrunc(sub_f, half, f"f2fp_{nm}_subh")
                    sub = IRBuilder.zext(
                        IRBuilder.bitcast(sub_h, i16, f"f2fp_{nm}_subi"),
                        i32, f"f2fp_{nm}_sub")
                    is_sub = IRBuilder.icmp_unsigned(
                        "==", e, self.ir.Constant(i32, 0), f"f2fp_{nm}_issub")
                    is_nan = IRBuilder.and_(
                        IRBuilder.icmp_unsigned(
                            "==", e, self.ir.Constant(i32, 0xF),
                            f"f2fp_{nm}_e15"),
                        IRBuilder.icmp_unsigned(
                            "==", m, self.ir.Constant(i32, 0x7),
                            f"f2fp_{nm}_m7"),
                        f"f2fp_{nm}_isnan")
                    mag = IRBuilder.select(
                        is_sub, sub,
                        IRBuilder.select(
                            is_nan, self.ir.Constant(i32, 0x7E00), norm,
                            f"f2fp_{nm}_nan_sel"),
                        f"f2fp_{nm}_mag")
                    h = IRBuilder.or_(mag, sgn, f"f2fp_{nm}_h")
                halves.append(h)
            result = IRBuilder.or_(
                halves[0],
                IRBuilder.shl(halves[1], self.ir.Constant(i32, 16),
                              "f2fp_hi_shift"),
                "f2fp_unpacked")
            self._set_val(IRBuilder, IRRegs, dest, result)
            return

        # F2FP.TF32.F32.PACK_B Rd, Rs — round one f32 to tf32 (19-bit: sign +
        # 8 exp + top 10 mantissa), zeroing the low 13 mantissa bits with
        # round-to-nearest-even.  This is PTX `cvt.rn.tf32.f32`.  TF32 is NOT in
        # _parse_f2f_types/_f2f_width (it has no LLVM float type), so without
        # this branch the op fell through to the standard F2F path as a 32->32
        # no-op (val = v1) and PASSED THE f32 THROUGH UNROUNDED — wrong for any
        # consumer (store/compare/non-HMMA use) that reads the bits directly.
        # The HW does the cheap uniform RN-even bias+mask on the 32-bit pattern:
        #   u' = u + 0x0FFF + ((u>>13)&1) ; result = u' & 0xFFFFE000
        # (Inf: low 13 already 0 -> bias+mask is identity; an ordinary qNaN
        # keeps its high mantissa bit after masking.)  Dest is typed INT32 by
        # the generic F2FP type rule, so emit the i32 bit pattern.
        if "TF32" in Inst.opcodes:
            v1 = _get_val(uses[0], "f2fp_tf32_src")
            u = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "f2fp_tf32_u")
            lsb = IRBuilder.and_(
                IRBuilder.lshr(u, self.ir.Constant(i32, 13), "f2fp_tf32_l13"),
                self.ir.Constant(i32, 1), "f2fp_tf32_lsb")
            rounded = IRBuilder.add(
                IRBuilder.add(u, self.ir.Constant(i32, 0x0FFF), "f2fp_tf32_r1"),
                lsb, "f2fp_tf32_r2")
            result = IRBuilder.and_(
                rounded, self.ir.Constant(i32, 0xFFFFE000), "f2fp_tf32")
            self._set_val(IRBuilder, IRRegs, dest, result)
            return

        # Standard F2F/F2FP (non-PACK_AB)
        op1 = uses[0]

        # Scalar BF16 endpoints are bit-level conversions — the generic
        # 16-bit path below maps width 16 to HalfType (f16, exp bias 15),
        # which neither encodes nor decodes bf16 (bias 127).
        if dest_type == "BF16":
            # F2F.BF16.F32[.RZ] Rd, Rs — bf16 bits land in Rd's LOW half
            # (corpus consumers read .H0 / SHF.L 0x10 the result back to
            # f32). Emitted as a half-typed BIT PATTERN, mirroring the
            # F2F.F16.F32 path's value kind, so downstream bit-level
            # consumers (PRMT, STS.U16, HADD2.BF16_V2) handle both alike.
            v1 = _get_val(op1, "f2f_src")
            src_float_type = float_type_map.get(src_width, self.ir.FloatType())
            if v1.type != src_float_type:
                v1 = self._insert_type_cast_if_needed(IRBuilder, v1, src_float_type, "f2f_src")
            if isinstance(v1.type, self.ir.DoubleType):
                v1 = IRBuilder.fptrunc(v1, self.ir.FloatType(), "f2f_d2f")
            bits = self._f32_to_bf16_bits_i32(IRBuilder, v1, "f2f_bf16",
                                              rz="RZ" in Inst.opcodes)
            val = IRBuilder.bitcast(
                IRBuilder.trunc(bits, i16, "f2f_bf16_i16"),
                self.ir.HalfType(), "f2f_bf16_h")
            self._set_val(IRBuilder, IRRegs, dest, val)
            return

        if src_type == "BF16":
            # F2F.F32.BF16 Rd, Rs — widen the bf16 in Rs's low half: bf16
            # is f32's top half, so bits << 16 IS the f32 (exact, no
            # rounding involved).
            v1 = _get_val(op1, "f2f_src")
            v1 = self._insert_type_cast_if_needed(IRBuilder, v1, i32, "f2f_bf16src")
            val = IRBuilder.bitcast(
                IRBuilder.shl(
                    IRBuilder.and_(v1, self.ir.Constant(i32, 0xFFFF),
                                   "f2f_bf16src_lo"),
                    self.ir.Constant(i32, 16), "f2f_bf16src_sh"),
                self.ir.FloatType(), "f2f_bf16src_f")
            if dest_width == 64:
                val = IRBuilder.fpext(val, self.ir.DoubleType(), "f2f_bf16src_ext")
            self._set_val(IRBuilder, IRRegs, dest, val)
            return

        target_float_type = float_type_map.get(dest_width, self.ir.HalfType())

        v1 = _get_val(op1, "f2f_src")
        src_float_type = float_type_map.get(src_width, self.ir.FloatType())
        if v1.type != src_float_type:
            # Use the size-aware cast helper rather than a bare bitcast —
            # bare bitcast i32→double fails the size check; the helper
            # routes through int widening when sizes differ.
            v1 = self._insert_type_cast_if_needed(IRBuilder, v1, src_float_type, "f2f_src")
        # Compare ACTUAL IR widths, not SASS-level src_width — src_width
        # for narrow FP types (E4M3=8, E5M2=8) maps to default f32 in
        # float_type_map so v1 has been promoted to f32. Comparing 8 < 16
        # would emit `fpext f32 to half`, which llc rejects.
        v1_bits = src_width
        if isinstance(v1.type, self.ir.HalfType):
            v1_bits = 16
        elif isinstance(v1.type, self.ir.FloatType):
            v1_bits = 32
        elif isinstance(v1.type, self.ir.DoubleType):
            v1_bits = 64

        if v1_bits < dest_width:
            val = IRBuilder.fpext(v1, target_float_type, "f2f_ext")
        elif v1_bits > dest_width:
            val = IRBuilder.fptrunc(v1, target_float_type, "f2f_trunc")
        else:
            val = v1
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("I2IP")
    def lift_i2ip(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        i32 = self.ir.IntType(32)
        src = _get_val(uses[0], "i2ip_src")
        pack_reg = _get_val(uses[1], "i2ip_pack")
        sel_op = uses[2]
        sel_val = _get_val(sel_op, "i2ip_sel")

        src = self._insert_type_cast_if_needed(IRBuilder, src, i32, "i2ip_src")
        pack_reg = self._insert_type_cast_if_needed(IRBuilder, pack_reg, i32, "i2ip_pack")
        sel_val = self._insert_type_cast_if_needed(IRBuilder, sel_val, i32, "i2ip_sel")

        dst_type = Inst.opcodes[1]
        src_type = Inst.opcodes[2]
        signed = dst_type.startswith("S")
        bits = int(dst_type[1:])

        byte_sel = int(sel_op.suffix) if sel_op.suffix and sel_op.suffix.isdigit() else 0
        shift = IRBuilder.lshr(sel_val, self.ir.Constant(i32, byte_sel), "i2ip_byte")
        shift = IRBuilder.and_(shift, self.ir.Constant(i32, 0xFF), "i2ip_shift")

        if "SAT" in Inst.opcodes:
            if signed:
                lo = -(1 << (bits - 1))
                hi = (1 << (bits - 1)) - 1
            else:
                lo = 0
                hi = (1 << bits) - 1
            cmp_lo = IRBuilder.icmp_signed("<", src, self.ir.Constant(i32, lo), "i2ip_cmp_lo")
            src = IRBuilder.select(cmp_lo, self.ir.Constant(i32, lo), src, "i2ip_sat_lo")
            cmp_hi = IRBuilder.icmp_signed(">", src, self.ir.Constant(i32, hi), "i2ip_cmp_hi")
            src = IRBuilder.select(cmp_hi, self.ir.Constant(i32, hi), src, "i2ip_sat_hi")

        mask_val = (1 << bits) - 1
        nibble = IRBuilder.and_(src, self.ir.Constant(i32, mask_val), "i2ip_nibble")
        nibble_shifted = IRBuilder.shl(nibble, shift, "i2ip_nib_shift")
        clear_mask = IRBuilder.shl(self.ir.Constant(i32, mask_val), shift, "i2ip_clr_mask")
        clear_mask = IRBuilder.not_(clear_mask, "i2ip_clr_inv")
        packed = IRBuilder.and_(pack_reg, clear_mask, "i2ip_cleared")
        result = IRBuilder.or_(packed, nibble_shifted, "i2ip_result")

        new_shift = IRBuilder.add(shift, self.ir.Constant(i32, bits), "i2ip_new_shift")
        upper = IRBuilder.shl(new_shift, self.ir.Constant(i32, byte_sel), "i2ip_upper")
        upper_mask = self.ir.Constant(i32, ~(0xFF << byte_sel) & 0xFFFFFFFF)
        result = IRBuilder.and_(result, upper_mask, "i2ip_clear_upper")
        result = IRBuilder.or_(result, upper, "i2ip_with_shift")

        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("IABS")
    def lift_iabs(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        v = _get_val(src, "iabs_src")
        res = self.run_intrinsic("abs", IRBuilder, [v], "iabs")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("VABSDIFF4")
    def lift_vabsdiff4(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src_a, src_b, src_c = Inst.get_uses()[0], Inst.get_uses()[1], Inst.get_uses()[2]

        modifiers = [tok.upper() for tok in Inst.opcodes[1:]]
        elem_type = "U8"
        if "S8" in modifiers:
            elem_type = "S8"
        elif "U8" in modifiers:
            elem_type = "U8"
        sat = "SAT" in modifiers

        i8 = self.ir.IntType(8)
        i32 = self.ir.IntType(32)

        v_a = _get_val(src_a, "vabsdiff4_a")
        v_b = _get_val(src_b, "vabsdiff4_b")
        v_c = _get_val(src_c, "vabsdiff4_c")

        v_a = self._insert_type_cast_if_needed(IRBuilder, v_a, i32, "vabsdiff4_a")
        v_b = self._insert_type_cast_if_needed(IRBuilder, v_b, i32, "vabsdiff4_b")
        v_c = self._insert_type_cast_if_needed(IRBuilder, v_c, i32, "vabsdiff4_c")

        mask_u8 = self.ir.Constant(i32, 0xFF)
        acc = self.ir.Constant(i32, 0)
        lane_absd = []

        for lane in range(4):
            shift_amt = self.ir.Constant(i32, lane * 8)
            a_shift = IRBuilder.lshr(v_a, shift_amt, f"vabsdiff4_a{lane}_shift")
            b_shift = IRBuilder.lshr(v_b, shift_amt, f"vabsdiff4_b{lane}_shift")

            if elem_type == "S8":
                a_lane_i8 = IRBuilder.trunc(a_shift, i8, f"vabsdiff4_a{lane}_i8")
                b_lane_i8 = IRBuilder.trunc(b_shift, i8, f"vabsdiff4_b{lane}_i8")
                a_lane = IRBuilder.sext(a_lane_i8, i32, f"vabsdiff4_a{lane}")
                b_lane = IRBuilder.sext(b_lane_i8, i32, f"vabsdiff4_b{lane}")

                diff = IRBuilder.sub(a_lane, b_lane, f"vabsdiff4_diff{lane}")
                absd = self.run_intrinsic("abs", IRBuilder, [diff], f"vabsdiff4_abs{lane}")
            else:
                a_lane = IRBuilder.and_(a_shift, mask_u8, f"vabsdiff4_a{lane}")
                b_lane = IRBuilder.and_(b_shift, mask_u8, f"vabsdiff4_b{lane}")

                ge = IRBuilder.icmp_unsigned(">=", a_lane, b_lane, f"vabsdiff4_ge{lane}")
                sub_ab = IRBuilder.sub(a_lane, b_lane, f"vabsdiff4_sub_ab{lane}")
                sub_ba = IRBuilder.sub(b_lane, a_lane, f"vabsdiff4_sub_ba{lane}")
                absd = IRBuilder.select(ge, sub_ab, sub_ba, f"vabsdiff4_abs{lane}")

            acc = IRBuilder.add(acc, absd, f"vabsdiff4_acc{lane}")
            lane_absd.append(absd)

        # `.ACC` (__vsad) sums the 4 byte abs-diffs into a scalar + accumulator;
        # plain VABSDIFF4 (__vabsdiffu4/s4) returns the diffs PACKED per byte.
        if "ACC" not in [m.upper() for m in Inst.opcodes[1:]]:
            packed = self.ir.Constant(i32, 0)
            for lane in range(4):
                byte = IRBuilder.and_(lane_absd[lane], mask_u8, f"vabsdiff4_pk_b{lane}")
                if lane:
                    byte = IRBuilder.shl(byte, self.ir.Constant(i32, lane * 8), f"vabsdiff4_pk_s{lane}")
                packed = IRBuilder.or_(packed, byte, f"vabsdiff4_pk{lane}")
            self._set_val(IRBuilder, IRRegs, dest, packed)
            return

        if sat:
            i64 = self.ir.IntType(64)
            acc_64 = IRBuilder.zext(acc, i64, "vabsdiff4_acc64")
            c_64 = IRBuilder.zext(v_c, i64, "vabsdiff4_c64")
            total_64 = IRBuilder.add(acc_64, c_64, "vabsdiff4_total64")
            max_64 = self.ir.Constant(i64, 0xFFFFFFFF)
            overflow = IRBuilder.icmp_unsigned(">", total_64, max_64, "vabsdiff4_sat_overflow")
            sat_64 = IRBuilder.select(overflow, max_64, total_64, "vabsdiff4_sat64")
            total = IRBuilder.trunc(sat_64, i32, "vabsdiff4_sat")
        else:
            total = IRBuilder.add(acc, v_c, "vabsdiff4")

        self._set_val(IRBuilder, IRRegs, dest, total)

    @lift_for("FRND", "UFRND")
    def lift_frnd(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        v = _get_val(src, "frnd_src")
        
        is_f64 = "F64" in Inst.opcodes
        suffix = "_f64" if is_f64 else "_f32"
        
        if "TRUNC" in Inst.opcodes:
            intrinsic_name = "trunc" + suffix
        elif "FLOOR" in Inst.opcodes:
            intrinsic_name = "floor" + suffix
        elif "CEIL" in Inst.opcodes:
            intrinsic_name = "ceil" + suffix
        else:
            intrinsic_name = "rint" + suffix
        
        res = self.run_intrinsic(intrinsic_name, IRBuilder, [v], "frnd")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("LOP3", "ULOP3", "PLOP3", "UPLOP3")
    def lift_lop3(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        if Inst.opcodes[1] != "LUT":
            raise UnsupportedInstructionException

        destPred = Inst.get_defs()[0] if len(Inst.get_defs()) > 1 else None
        destReg = Inst.get_defs()[-1]
        src1, src2, src3 = Inst.get_uses()[0], Inst.get_uses()[1], Inst.get_uses()[2]

        lut = Inst.get_uses()[3]
        src4 = Inst.get_uses()[4]

        a = _get_val(src1, "lop3_a")
        b = _get_val(src2, "lop3_b")
        c = _get_val(src3, "lop3_c")
        q = _get_val(src4, "lop3_q")

        lut_val = lut.immediate_value

        i32 = self.ir.IntType(32)
        a = self._insert_type_cast_if_needed(IRBuilder, a, i32, "lop3_a_cast")
        b = self._insert_type_cast_if_needed(IRBuilder, b, i32, "lop3_b_cast")
        c = self._insert_type_cast_if_needed(IRBuilder, c, i32, "lop3_c_cast")

        # PLOP3/UPLOP3 operate on PREDICATES (i1 carried as i32 0/1), not 32-bit
        # words: the negated term must be a single-bit NOT (a^1), and the inputs
        # masked to bit 0. A full-width `a^-1` (~1 = 0xFFFFFFFE) leaves spurious
        # high bits set, which the predicate's final `r != 0` test then catches —
        # flipping a "false" LUT term to true (fpc: PLOP3 0xf1 with a=0,b=1,c=1
        # wrongly yielded P=true -> misclassification). LOP3/ULOP3 stay full-width
        # (true bitwise ops; their predicate output is genuinely `result != 0`).
        is_pred_logic = Inst.opcodes[0] in ("PLOP3", "UPLOP3")
        if is_pred_logic:
            one = self.ir.Constant(i32, 1)
            a = IRBuilder.and_(a, one, "plop3_a_bit")
            b = IRBuilder.and_(b, one, "plop3_b_bit")
            c = IRBuilder.and_(c, one, "plop3_c_bit")
        neg = self.ir.Constant(i32, 1 if is_pred_logic else -1)
        na = IRBuilder.xor(a, neg)
        nb = IRBuilder.xor(b, neg)
        nc = IRBuilder.xor(c, neg)

        zero = self.ir.Constant(i32, 0)
        r = zero

        for idx in range(8):
            if (lut_val >> idx) & 1 == 0:
                continue

            a_bit = (idx >> 2) & 1
            b_bit = (idx >> 1) & 1
            c_bit = idx & 1

            va = a if a_bit else na
            vb = b if b_bit else nb
            vc = c if c_bit else nc

            t_ab = IRBuilder.and_(va, vb)
            t_term = IRBuilder.and_(t_ab, vc)

            if r is zero:
                r = t_term
            else:
                r = IRBuilder.or_(r, t_term)

        target_ty = destReg.get_ir_type(self)
        if destReg.is_writable_reg:
            result = self._insert_type_cast_if_needed(IRBuilder, r, target_ty, "lop3_result") if r.type != target_ty else r
            self._set_val(IRBuilder, IRRegs, destReg, result)

        if destPred and destPred.is_writable_reg:
            tmp = IRBuilder.icmp_signed("!=", r, zero, "lop3_pred_cmp")
            # q (4th source) is a predicate operand; _get_val may have widened
            # it to i32 if the source pseudo-register was typed as int (e.g.,
            # !PT constant fold). Normalize to i1 so or_ types match.
            q = self._normalize_i1_in_curr_bb(IRBuilder, q, "lop3_q")
            self._set_val(IRBuilder, IRRegs, destPred, IRBuilder.or_(tmp, q, "lop3_p"))

    @lift_for("MOVM")
    def lift_movm(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "movm")
        if "MT88" in Inst.opcodes:
            i32 = self.ir.IntType(32)
            val = self._insert_type_cast_if_needed(IRBuilder, val, i32, "movm_src")
            asm_ty = self.ir.FunctionType(i32, [i32])
            # `movmatrix.sync.aligned` is warp-collective: the result on each
            # lane depends on the input value on every other lane in the warp.
            # Without `side_effect=True`, LLVM treats this as a pure function
            # of its single i32 input and is free to reorder it across other
            # collective ops (other movmatrix calls, the surrounding LDS that
            # populated `val`, etc.) — which corrupts the cross-lane
            # transpose. side_effect=True keeps the call pinned to its
            # original program-order position. This matters for tensor /
            # wmma kernels where each HMMA.1688.F32 reads B from a register
            # that was just transposed by movmatrix; reordering produces
            # numerically-wrong results.
            inline_asm = self.ir.InlineAsm(
                asm_ty,
                "movmatrix.sync.aligned.m8n8.trans.b16 $0, $1;",
                "=r,r",
                side_effect=True,
            )
            val = IRBuilder.call(inline_asm, [val], "movm_trans")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("HMMA")
    def lift_hmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        size = Inst.opcodes[1]
        accum_type = Inst.opcodes[2]

        if size == "884":
            step_opcodes = [opc for opc in Inst.opcodes if opc.startswith("STEP")]
            step = int(step_opcodes[0][-1]) if step_opcodes else 0
            uses = Inst.get_uses()
            dests = Inst.get_defs()
            i32 = self.ir.IntType(32)
            f32 = self.ir.FloatType()
            v2f16 = self.ir.VectorType(self.ir.HalfType(), 2)

            a0 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[0]), i32, "hmma884_a0")
            a1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[1]), i32, "hmma884_a1")
            b0 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[2]), i32, "hmma884_b0")
            b1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[3]), i32, "hmma884_b1")
            a0 = IRBuilder.bitcast(a0, v2f16, "hmma884_a0_v2f16")
            a1 = IRBuilder.bitcast(a1, v2f16, "hmma884_a1_v2f16")
            b0 = IRBuilder.bitcast(b0, v2f16, "hmma884_b0_v2f16")
            b1 = IRBuilder.bitcast(b1, v2f16, "hmma884_b1_v2f16")

            if accum_type == "F32":
                c0 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[4]), f32, "hmma884_c0")
                c1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[5]), f32, "hmma884_c1")
                zero_f = self.ir.Constant(f32, 0.0)
                c_vals = [zero_f] * 8
                c_vals[2 * step] = c0
                c_vals[2 * step + 1] = c1

                args = [a0, a1, b0, b1] + c_vals
                val = self.run_intrinsic("hmma884f32", IRBuilder, args, "hmma884_call")
                self._set_val(IRBuilder, IRRegs, dests[0], IRBuilder.extract_value(val, 2 * step, "hmma884_d0"))
                self._set_val(IRBuilder, IRRegs, dests[1], IRBuilder.extract_value(val, 2 * step + 1, "hmma884_d1"))
            else:
                c0_raw = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[4]), i32, "hmma884_c0")
                c1_raw = self._insert_type_cast_if_needed(IRBuilder, _get_val(uses[5]), i32, "hmma884_c1")
                c0 = IRBuilder.bitcast(c0_raw, v2f16, "hmma884_c0_v2f16")
                c1 = IRBuilder.bitcast(c1_raw, v2f16, "hmma884_c1_v2f16")
                zero_v = self.ir.Constant(v2f16, [self.ir.Constant(self.ir.HalfType(), 0.0)] * 2)
                c_vals = [zero_v] * 4
                c_vals[step] = c0
                if step + 1 < 4:
                    c_vals[step + 1] = c1

                args = [a0, a1, b0, b1] + c_vals
                val = self.run_intrinsic("hmma884f16", IRBuilder, args, "hmma884_call")
                d0_v = IRBuilder.extract_value(val, step, "hmma884_d0_v2f16")
                self._set_val(IRBuilder, IRRegs, dests[0], IRBuilder.bitcast(d0_v, i32, "hmma884_d0"))
                d1_idx = min(step + 1, 3)
                d1_v = IRBuilder.extract_value(val, d1_idx, "hmma884_d1_v2f16")
                self._set_val(IRBuilder, IRRegs, dests[1], IRBuilder.bitcast(d1_v, i32, "hmma884_d1"))
            return

        if size not in ("1688", "16816"):
            raise UnsupportedInstructionException(f"Unsupported HMMA shape: {size}")

        uses = Inst.get_uses()
        dests = Inst.get_defs()

        # TF32-input form (HMMA.1688.F32.TF32): one tf32 value per FULL
        # 32-bit register, not packed 16-bit pairs — A:4 + B:2 + C:4 = 10
        # uses (cute MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>). Route to a
        # dedicated intrinsic taking scalar f32 args; the runtime masks the
        # low 13 mantissa bits (tf32 precision) before the f32 FMA.
        if "TF32" in Inst.opcodes:
            if size != "1688" or accum_type != "F32":
                raise UnsupportedInstructionException(
                    f"Unsupported TF32 HMMA: {'.'.join(Inst.opcodes)}")
            f32 = self.ir.FloatType()
            args = []
            for i, u in enumerate(uses):
                v = _get_val(u, f"hmma_arg_{i}")
                if v.type != f32:
                    v = self._insert_type_cast_if_needed(
                        IRBuilder, v, f32, f"hmma_arg_{i}")
                args.append(v)
            val = self.run_intrinsic("hmma1688f32_tf32", IRBuilder, args,
                                     "hmma_call")
            for i in range(len(dests)):
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.extract_value(val, i, f"hmma_res_{i}"))
            return

        # 1688: 3 v2f16 operands (A:2 + B:1), then 4 f32 accumulators => 7 uses
        # 16816: 6 v2f16 operands (A:4 + B:2), then 4 f32 accumulators => 10 uses
        n_fp16_regs = 6 if size == "16816" else 3
        intrinsic_prefix = f"hmma{size}"
        # BF16-input form keeps the F32 accumulator but packs A/B as
        # <2 x bfloat> instead of <2 x half>. SASS opcode is e.g.
        # HMMA.16816.F32.BF16. We route to a distinct intrinsic so the
        # function signature matches the element type.
        is_bf16_input = "BF16" in Inst.opcodes

        if accum_type == "F32":
            elem_ty = self._half2_elem_ty(is_bf16_input)
            v2packed = self.ir.VectorType(elem_ty, 2)
            f32 = self.ir.FloatType()
            args = []
            for i, u in enumerate(uses):
                if i < n_fp16_regs:
                    v = self._get_val_half2(
                        IRBuilder, _get_val, u,
                        f"hmma_arg_{i}", is_bf16=is_bf16_input,
                    )
                else:
                    v = _get_val(u, f"hmma_arg_{i}")
                    if v.type != f32:
                        v = self._insert_type_cast_if_needed(
                            IRBuilder, v, f32, f"hmma_arg_{i}"
                        )
                args.append(v)
            suffix = "f32_bf16" if is_bf16_input else "f32"
            val = self.run_intrinsic(f"{intrinsic_prefix}{suffix}", IRBuilder, args, "hmma_call")
            for i in range(len(dests)):
                self._set_val(IRBuilder, IRRegs, dests[i], IRBuilder.extract_value(
                    val, i, f"hmma_res_{i}"
                ))
        elif accum_type == "F16":
            v2f16 = self.ir.VectorType(self.ir.HalfType(), 2)
            i32 = self.ir.IntType(32)
            args = []
            for i, u in enumerate(uses):
                v = _get_val(u, f"hmma_arg_{i}")
                vbits = v.type.width if v.type.intrinsic_name[0] == 'i' else int(v.type.intrinsic_name[1:])
                if vbits != 32:
                    v = self._insert_type_cast_if_needed(IRBuilder, v, i32, f"hmma_arg_{i}")
                if v.type != v2f16:
                    v = IRBuilder.bitcast(v, v2f16, f"hmma_arg_{i}_v2f16")
                args.append(v)
            val = self.run_intrinsic(f"{intrinsic_prefix}f16", IRBuilder, args, "hmma_call")
            for i in range(len(dests)):
                res = IRBuilder.extract_value(val, i, f"hmma_res_{i}")
                res = IRBuilder.bitcast(res, self.ir.IntType(32), f"hmma_res_{i}_i32")
                self._set_val(IRBuilder, IRRegs, dests[i], res)
        else:
            raise UnsupportedInstructionException(f"Unsupported HMMA type: {accum_type}")

    @lift_for("DMMA")
    def lift_dmma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """DMMA — F64 tensor-core MMA.

        8x8x4: lane-0 scalar FMA stub via the dmma884f64 intrinsic (legacy).

        16x8x16 (sm90 dgemm): faithful warp-true compute via the runtime
        wrapper call dmma_16816_f64. The wrapper rendezvouses the 32 lanes
        (warp_group_gather, same machinery as ballot/match), reconstructs
        the full 16x16 A / 16x8 B from everyone's fragments, and each lane
        computes its own 4 D values. Pointer-out ABI (no struct return —
        4xf64 would be sret-classified and is easy to get wrong; cf. the
        vote_* i1-vs-i8 SysV bug).
        """
        shape = Inst.opcodes[1] if len(Inst.opcodes) > 1 else ""
        uses = Inst.get_uses()
        dests = Inst.get_defs()
        f64 = self.ir.DoubleType()

        if shape == "8x8x4":
            # m8n8k4 F64: warp-true compute via the runtime wrapper dmma_884_f64,
            # same warp-rendezvous design as the 16x8x16 path. The atom is a
            # warp collective (8x8 D = 2/lane, 8x4 A / 8x4 B = 1/lane) so the
            # earlier per-lane scalar-FMA stub could not reconstruct it. Fragments
            # per cute MMA_Traits<SM80_8x8x4_F64F64F64F64_TN>: 1 A, 1 B, 2 C, 2 D.
            if len(uses) != 4 or len(dests) != 2:
                raise UnsupportedInstructionException(
                    f"DMMA.8x8x4 expects 4 uses / 2 defs, got {len(uses)}/{len(dests)}")
            args = []
            for i, u in enumerate(uses):
                v = _get_val(u, f"dmma_arg_{i}")
                if v.type != f64:
                    v = self._insert_type_cast_if_needed(IRBuilder, v, f64, f"dmma_arg_{i}")
                args.append(v)
            fname = "dmma_884_f64"
            func = self.llvm_module.globals.get(fname)
            if func is None:
                ptr_f64 = self.ir.PointerType(f64)
                func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                               [ptr_f64] + [f64] * 4, False)
                func = self.ir.Function(self.llvm_module, func_ty, fname)
            d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(f64, 2),
                                        "dmma884_d_out")
            d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(f64), "dmma884_d_ptr")
            IRBuilder.call(func, [d_ptr] + args)
            for i in range(len(dests)):
                ep = IRBuilder.gep(d_slot,
                                   [self.ir.Constant(self.ir.IntType(32), 0),
                                    self.ir.Constant(self.ir.IntType(32), i)],
                                   name=f"dmma884_d{i}_ep")
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.load(ep, f"dmma_res_{i}"))
            return

        if shape != "16x8x16":
            raise UnsupportedInstructionException(f"Unsupported DMMA shape: {shape}")

        # 8 A + 4 B + 4 C per-lane f64 fragments.
        if len(uses) != 16 or len(dests) != 4:
            raise UnsupportedInstructionException(
                f"DMMA.16x8x16 expects 16 uses / 4 defs, got {len(uses)}/{len(dests)}")
        args = []
        for i, u in enumerate(uses):
            v = _get_val(u, f"dmma_arg_{i}")
            if v.type != f64:
                v = self._insert_type_cast_if_needed(IRBuilder, v, f64, f"dmma_arg_{i}")
            args.append(v)

        fname = "dmma_16816_f64"
        func = self.llvm_module.globals.get(fname)
        if func is None:
            ptr_f64 = self.ir.PointerType(f64)
            func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                           [ptr_f64] + [f64] * 16, False)
            func = self.ir.Function(self.llvm_module, func_ty, fname)
        d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(f64, 4),
                                    "dmma_d_out")
        d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(f64), "dmma_d_ptr")
        IRBuilder.call(func, [d_ptr] + args)
        for i in range(len(dests)):
            ep = IRBuilder.gep(d_slot,
                               [self.ir.Constant(self.ir.IntType(32), 0),
                                self.ir.Constant(self.ir.IntType(32), i)],
                               name=f"dmma_d{i}_ep")
            self._set_val(IRBuilder, IRRegs, dests[i],
                          IRBuilder.load(ep, f"dmma_res_{i}"))

    @lift_for("IMMA")
    def lift_imma(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # IMMA.SP.<shape>.S8.S8[.SAT] is sparse 2:4 IMMA; the metadata register
        # and selector immediate at the tail of get_uses() are dropped on
        # x86 single-thread (dense compute).
        is_sparse = len(Inst.opcodes) > 1 and Inst.opcodes[1] == "SP"
        base = 2 if is_sparse else 1
        shape = Inst.opcodes[base]
        src_type_a = Inst.opcodes[base + 1] if len(Inst.opcodes) > base + 1 else ""
        src_type_b = Inst.opcodes[base + 2] if len(Inst.opcodes) > base + 2 else ""
        sat = "SAT" in Inst.opcodes

        uses = Inst.get_uses()
        dests = Inst.get_defs()
        i32 = self.ir.IntType(32)

        # Sparse form appends metadata + immediate; drop the trailing two
        # operands (1 metadata reg + 1 selector immediate) before casting.
        if is_sparse:
            uses_for_args = uses[:-2]
        else:
            uses_for_args = uses
        args = [self._insert_type_cast_if_needed(IRBuilder, _get_val(uses_for_args[i], f"imma_arg_{i}"), i32, f"imma_arg_{i}") for i in range(len(uses_for_args))]

        a_ty = src_type_a.lower()
        b_ty = src_type_b.lower()
        sat_str = ".satfinite" if sat else ""
        sp_str = "_sparse" if is_sparse else ""

        # IMMA.8816 (m8n8k16, int8->int32) — warp-true compute via the runtime
        # wrapper imma_8816_s8_warp, same warp-rendezvous design as the
        # DMMA.8x8x4 path. The atom is a warp collective (8x8 D = 2/lane, 8x16 A
        # / 8x16 B = 1 packed-4xs8 reg/lane) so the earlier per-lane C->D
        # identity stub dropped the A*B product entirely (08_turing &c computed
        # D = C). Fragments per cute MMA_Traits<SM75_8x8x16_S32S8S8S32_TN>: 1 A
        # reg (4 packed s8), 1 B reg (4 packed s8), 2 C (s32), 2 D (s32).
        # Scoped to dense s8 8816 only — every other IMMA shape/type keeps the
        # existing run_intrinsic path below.
        if shape == "8816" and not is_sparse and a_ty == "s8" and b_ty == "s8":
            if len(uses) != 4 or len(dests) != 2:
                raise UnsupportedInstructionException(
                    f"IMMA.8816 expects 4 uses / 2 defs, got {len(uses)}/{len(dests)}")
            # args already cast to i32 above (order = A, B, C0, C1).
            fname = "imma_8816_s8_warp"
            func = self.llvm_module.globals.get(fname)
            if func is None:
                ptr_i32 = self.ir.PointerType(i32)
                func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                               [ptr_i32] + [i32] * 4, False)
                func = self.ir.Function(self.llvm_module, func_ty, fname)
            d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(i32, 2),
                                        "imma8816_d_out")
            d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(i32),
                                      "imma8816_d_ptr")
            IRBuilder.call(func, [d_ptr] + args)
            for i in range(len(dests)):
                ep = IRBuilder.gep(d_slot,
                                   [self.ir.Constant(self.ir.IntType(32), 0),
                                    self.ir.Constant(self.ir.IntType(32), i)],
                                   name=f"imma8816_d{i}_ep")
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.load(ep, f"imma_res_{i}"))
            return

        # IMMA.16832 (m16n8k32, int8->int32) — warp-true compute via the runtime
        # wrapper imma_16832_s8_warp, the larger-shape sibling of the IMMA.8816
        # path (same warp-rendezvous design). The atom is a warp collective
        # (16x8 D = 4/lane, 16x32 A = 4 packed-4xs8 regs/lane, 8x32 B = 2 packed-
        # 4xs8 regs/lane) so the earlier per-lane C->D identity stub dropped the
        # A*B product entirely (13_two_tensor_op_fusion s8 &c computed D = C).
        # Fragments per cute MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>: 4 A regs, 2 B
        # regs, 4 C (s32), 4 D (s32). Scoped to dense s8 16832 only — every other
        # IMMA shape/type keeps the existing run_intrinsic path below.
        if shape == "16832" and not is_sparse and a_ty == "s8" and b_ty == "s8":
            if len(uses) != 10 or len(dests) != 4:
                raise UnsupportedInstructionException(
                    f"IMMA.16832 expects 10 uses / 4 defs, got {len(uses)}/{len(dests)}")
            # args already cast to i32 above (order = A0..A3, B0..B1, C0..C3).
            fname = "imma_16832_s8_warp"
            func = self.llvm_module.globals.get(fname)
            if func is None:
                ptr_i32 = self.ir.PointerType(i32)
                func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                               [ptr_i32] + [i32] * 10, False)
                func = self.ir.Function(self.llvm_module, func_ty, fname)
            d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(i32, 4),
                                        "imma16832_d_out")
            d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(i32),
                                      "imma16832_d_ptr")
            IRBuilder.call(func, [d_ptr] + args)
            for i in range(len(dests)):
                ep = IRBuilder.gep(d_slot,
                                   [self.ir.Constant(self.ir.IntType(32), 0),
                                    self.ir.Constant(self.ir.IntType(32), i)],
                                   name=f"imma16832_d{i}_ep")
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.load(ep, f"imma_res_{i}"))
            return

        # IMMA.SP.16864 INT4 disambiguation. The int4 m16n8k128 sparse atom
        # (CUTLASS SparseMma<GemmShape<16,8,128>,32,int4b_t,...>, PTX
        # mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite) lowers to
        # SASS that is BYTE-IDENTICAL to the int8 m16n8k64 atom: the SAME mnemonic
        # IMMA.SP.16864.S8.S8, no .S4 token, no distinct shape — nvcc emits a PAIR
        # of these K=64 sub-atoms per logical int4 atom. The dtype/shape survive
        # ONLY in the kernel-symbol (mangled) name, so the int8-vs-int4 split MUST
        # come from there. The cutlass int4-sparse kernel name carries both
        # `integer_subbyteILi4` (int4 element) and `ILi16ELi8ELi128EE` (the
        # SparseMma<GemmShape<16,8,128>> instruction shape). When present, route the
        # sparse atom to the int4 wrapper imma_sp_s4_warp (int4 unpack: 8 nibbles/reg
        # + the int4 2:4 metadata, cutlass host_uncompress.h int4 rule) instead of
        # the int8 imma_sp16864_s8_warp (which would int8-unpack int4 data → wrong).
        # Same call shape as the int8 wrapper (13 i32 args). Scoped to the int4-
        # tagged sparse-16864 kernel ONLY; genuine int8 kernels lack the markers.
        #
        # The logical int4 m16n8k128 atom lowers to a PAIR of these K=64 SASS
        # sub-atoms whose D the kernel SUMS (atom2.D += atom1.D). Route EACH sub-atom
        # to imma_sp_s4_k64_warp — the NATIVE K=64 int4 sparse op (the proven int8
        # imma_sp16864 K=64 thread map reading int4 low-nibbles); the existing SASS
        # IMAD.IADD sum then forms the K=128 product, no pairing recognizer. ptxas
        # splits the logical fragments by nibble parity into the two sub-atoms'
        # regs (derived and verified for all 32 lanes without a GPU). The
        # full-K128 imma_sp_s4_warp is bit-exact ONLY on LOGICAL fragments; a SASS
        # sub-atom carries the ptxas-repacked native-K64 fragments, so routing it to
        # the K128 wrapper double-counts (128/128 wrong) — imma_sp_s4_k64_warp is the
        # per-sub-atom-correct route.
        _kname = getattr(getattr(self, "curr_function", None), "name", "") or ""
        _is_int4_sparse = ("integer_subbyteILi4" in _kname
                           and "ILi16ELi8ELi128EE" in _kname)
        if shape == "16864" and is_sparse and a_ty == "s8" and b_ty == "s8" and _is_int4_sparse:
            if len(uses) != 14 or len(dests) != 4:
                raise UnsupportedInstructionException(
                    f"IMMA.SP int4 (m16n8k128) expects 14 uses / 4 defs, got {len(uses)}/{len(dests)}")
            # args = uses[:-2] (A0..3, B0..3, C0..3) already cast to i32 above.
            # Append the metadata reg (uses[-2]); skip the selector immediate.
            e_val = self._insert_type_cast_if_needed(
                IRBuilder, _get_val(uses[-2], "imma_sps4_meta"), i32, "imma_sps4_meta")
            call_args = args + [e_val]   # 12 frag args + 1 E = 13 i32 args
            fname = "imma_sp_s4_k64_warp"
            func = self.llvm_module.globals.get(fname)
            if func is None:
                ptr_i32 = self.ir.PointerType(i32)
                func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                               [ptr_i32] + [i32] * 13, False)
                func = self.ir.Function(self.llvm_module, func_ty, fname)
            d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(i32, 4),
                                        "immasps4_d_out")
            d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(i32),
                                      "immasps4_d_ptr")
            IRBuilder.call(func, [d_ptr] + call_args)
            for i in range(len(dests)):
                ep = IRBuilder.gep(d_slot,
                                   [self.ir.Constant(self.ir.IntType(32), 0),
                                    self.ir.Constant(self.ir.IntType(32), i)],
                                   name=f"immasps4_d{i}_ep")
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.load(ep, f"imma_res_{i}"))
            return

        # IMMA.SP.16864 (m16n8k64, int8->int32, 2:4 structured sparse) — warp-true
        # compute via the runtime wrapper imma_sp16864_s8_warp. CUTLASS
        # 15_ampere_sparse_tensorop_gemm emits this (PTX mma.sp.sync.aligned.
        # m16n8k64.row.col.s32.s8.s8.s32.satfinite). A is SPARSE-COMPRESSED (32 of
        # every 64 K-cols/row kept) and the metadata operand E selects which 2 of
        # every 4 dense K positions per group the kept values occupy. The earlier
        # per-lane C->D identity stub dropped the A*B product entirely (D = C).
        # Like the dense 8816 path this is a warp collective: 4 packed-4xs8 A regs,
        # 4 packed-4xs8 B regs, 4 C (s32), 4 D (s32) per lane + a 1-reg metadata E.
        # get_uses() = [A0..3, B0..3, C0..3, meta_reg, selector_imm] (14); the
        # wrapper takes A0..3, B0..3, C0..3, E (selector_imm is always 0x0 and is
        # dropped). Scoped to sparse s8 16864 only (NOT the int4 m16n8k128 atom,
        # handled by the int4 branch above).
        if shape == "16864" and is_sparse and a_ty == "s8" and b_ty == "s8":
            if len(uses) != 14 or len(dests) != 4:
                raise UnsupportedInstructionException(
                    f"IMMA.SP.16864 expects 14 uses / 4 defs, got {len(uses)}/{len(dests)}")
            # args = uses[:-2] (A0..3, B0..3, C0..3) already cast to i32 above.
            # Append the metadata reg (uses[-2]); skip the selector immediate.
            e_val = self._insert_type_cast_if_needed(
                IRBuilder, _get_val(uses[-2], "imma_sp_meta"), i32, "imma_sp_meta")
            call_args = args + [e_val]   # 12 frag args + 1 E = 13 i32 args
            fname = "imma_sp16864_s8_warp"
            func = self.llvm_module.globals.get(fname)
            if func is None:
                ptr_i32 = self.ir.PointerType(i32)
                func_ty = self.ir.FunctionType(self.ir.VoidType(),
                                               [ptr_i32] + [i32] * 13, False)
                func = self.ir.Function(self.llvm_module, func_ty, fname)
            d_slot = self._entry_alloca(IRBuilder, self.ir.ArrayType(i32, 4),
                                        "immasp_d_out")
            d_ptr = IRBuilder.bitcast(d_slot, self.ir.PointerType(i32),
                                      "immasp_d_ptr")
            IRBuilder.call(func, [d_ptr] + call_args)
            for i in range(len(dests)):
                ep = IRBuilder.gep(d_slot,
                                   [self.ir.Constant(self.ir.IntType(32), 0),
                                    self.ir.Constant(self.ir.IntType(32), i)],
                                   name=f"immasp_d{i}_ep")
                self._set_val(IRBuilder, IRRegs, dests[i],
                              IRBuilder.load(ep, f"imma_res_{i}"))
            return

        shape_map = {
            "8816": "m8n8k16",
            "8832": "m8n8k32",
            "16816": "m16n8k16",
            "16832": "m16n8k32",
            "16864": "m16n8k64",
        }
        if shape not in shape_map:
            raise UnsupportedInstructionException(f"Unsupported IMMA shape: {shape}")

        # intrinsic suffix: when A==B, just one type; otherwise both
        type_suffix = f".{a_ty}" if a_ty == b_ty else f".{a_ty}.{b_ty}"
        intrinsic_name = f"imma_{shape_map[shape]}{sat_str.replace('.', '_')}{type_suffix.replace('.', '_')}{sp_str}"
        val = self.run_intrinsic(intrinsic_name, IRBuilder, args, "imma_call")

        for i in range(len(dests)):
            self._set_val(IRBuilder, IRRegs, dests[i], IRBuilder.extract_value(
                val, i, f"imma_res_{i}"
            ))

    @lift_for("DEPBAR")
    def lift_depbar(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass
    
    @lift_for("MEMBAR")
    def lift_membar(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pass

    @lift_for("ULDC", "ULDC64", "LDC", "LDCU", "LDCU64")
    def lift_uldc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        is_64 = "64" in Inst.opcodes
        # LDC.U8/S8/U16/S16 load a 1- or 2-byte constant then zero/sign-extend.
        # Load at the exact width (a 32-bit load at a non-4B-aligned const_mem
        # offset faults), mirroring lift_ldg/lift_lds.
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
        if src.is_const_mem:
            bank = src.const_mem_bank
            # Use exact width: 64-bit for .64 instructions, 32-bit otherwise.
            # TypeAnalysis may over-propagate Int64 to 32-bit LDC sources.
            if narrow_ty is not None:
                load_ty = narrow_ty
            else:
                load_ty = self.ir.IntType(64) if is_64 else self.ir.IntType(32)
            if src.is_reg and src.reg and src.reg not in ("RZ", "URZ"):
                # Register-indexed load, e.g. `LDC.S8 Rd, c[0x3][R0]`: the byte
                # offset is dynamic (R0[+imm]), so index const_mem at runtime.
                # (A zero-register index, `c[bank][RZ]`, is just a static load and
                # falls through to the offset path below.)
                idx_val = IRRegs[src.reg]
                if src.offset_value is not None:
                    idx_val = IRBuilder.add(
                        idx_val, self.ir.Constant(idx_val.type, int(src.offset_value)),
                        "ldc_idx")
                val = self._load_from_const_mem_dynamic(IRBuilder, bank, idx_val, load_ty, "ldc_const")
            else:
                byte_offset = int(src.offset_value) if src.offset_value is not None else 0
                val = self._load_from_const_mem(IRBuilder, bank, byte_offset, load_ty, "ldc_const")
            if narrow_ty is not None:
                dest_ty = dest.get_ir_type(self)
                if isinstance(dest_ty, self.ir.IntType) and dest_ty.width > narrow_ty.width:
                    if is_signed:
                        val = IRBuilder.sext(val, dest_ty, "ldc_sext")
                    else:
                        val = IRBuilder.zext(val, dest_ty, "ldc_zext")
        else:
            val = _get_val(src, "ldc_const")
        self._set_val(IRBuilder, IRRegs, dest, val)

    @lift_for("FSETP", "DSETP", "UFSETP")
    def lift_fsetp(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest1 = Inst.get_defs()[0]
        dest2 = Inst.get_defs()[1]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        src = Inst.get_uses()[2]

        # Cast both operands to the compare type before fcmp (they may arrive as
        # i32 carriers from a non-FP producer). DSETP is double precision — using
        # f32 here would truncate 24 mantissa bits and corrupt downstream f64.
        is_dsetp = Inst.opcodes[0] == "DSETP"
        cmp_ty = self.ir.DoubleType() if is_dsetp else self.ir.FloatType()
        val1 = self._insert_type_cast_if_needed(IRBuilder, _get_val(a, "fsetrp_lhs"), cmp_ty, "fsetrp_lhs")
        val2 = self._insert_type_cast_if_needed(IRBuilder, _get_val(b, "fsetrp_rhs"), cmp_ty, "fsetrp_rhs")
        val3 = _get_val(src, "fsetrp_src")

        cmp1 = Inst.opcodes[1]
        # Handle optional FTZ flag: FSETP.NEU.FTZ.AND vs FSETP.NEU.AND
        if Inst.opcodes[2] == "FTZ":
            cmp2 = Inst.opcodes[3]
        else:
            cmp2 = Inst.opcodes[2]

        isUnordered = "U" in cmp1

        funcs_map = {
            "EQ": "==",
            "NE": "!=",
            "LT": "<",
            "LE": "<=",
            "GT": ">",
            "GE": ">=",
            "NEU": "!=",
            "LTU": "<",
            "LEU": "<=",
            "GTU": ">",
            "GEU": ">=",
            "EQU": "==",
            "MAX": "ogt",
            "MIN": "olt",
            "NAN": "uno",
            "NUM": "ord",
        }

        cmp2_map = {
            "AND": IRBuilder.and_,
            "OR": IRBuilder.or_,
            "XOR": IRBuilder.xor,
        }

        if isUnordered:
            r = IRBuilder.fcmp_unordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")
        else:
            r = IRBuilder.fcmp_ordered(funcs_map[cmp1], val1, val2, "fsetrp_cmp1")

        cmp2_func = cmp2_map.get(cmp2)
        if not cmp2_func:
            raise UnsupportedInstructionException

        # Normalize val3 to i1 (same as lift_isetp). _get_val may return
        # it as a wider int if the source pseudo-register was typed as i32
        # during seeding; and_/or_/xor_ requires matching operand types.
        i1 = self.ir.IntType(1)
        if val3.type != i1:
            if isinstance(val3.type, self.ir.IntType):
                zero = self.ir.Constant(val3.type, 0)
                val3 = IRBuilder.icmp_unsigned("!=", val3, zero, "fsetrp_src_to_i1")
            else:
                val3 = IRBuilder.bitcast(val3, i1, "fsetrp_src_bitcast_i1")

        r1 = cmp2_func(r, val3, "fsetrp_cmp2")

        if not dest1.is_pt:
            self._set_val(IRBuilder, IRRegs, dest1, r1)

        if not dest2.is_pt:
            if cmp1 in ("MIN", "MAX"):
                # DSETP.MIN/MAX produce a DUAL predicate where P1 is the
                # UNORDERED (NaN) flag, NOT !P0. The compiler uses P1 to guard
                # IEEE NaN-quiet-bit canonicalization of the fmin/fmax result
                # (`@P1 LOP3 Rd, Rhi, 0x80000, RZ, 0xfc` = OR in mantissa bit 51).
                # Setting P1=!P0 (the plain-SETP rule) fires that OR for ordered
                # non-NaN inputs and corrupts the result's high word (f64 fmin
                # gave a-with-bit51-set: -11.875 -> -15.875). P1 = isnan(a)||isnan(b).
                uno = IRBuilder.fcmp_ordered("uno", val1, val2, "fsetrp_uno")
                r2 = cmp2_func(uno, val3, "fsetrp_uno_cmb")
            else:
                r2 = IRBuilder.not_(r1, "fsetrp_not")
            self._set_val(IRBuilder, IRRegs, dest2, r2)

    @lift_for("ISETP", "ISETP64", "UISETP")
    def lift_isetp(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest1 = Inst.get_defs()[0]
        dest2 = Inst.get_defs()[1]
        a, b = Inst.get_uses()[0], Inst.get_uses()[1]
        src = Inst.get_uses()[2]

        val1 = _get_val(a, "isetrp_lhs")
        val2 = _get_val(b, "isetrp_rhs")
        val3 = _get_val(src, "isetrp_src")

        cmp1 = Inst.opcodes[1]
        # Parse modifier chain: ISETP.cmp.[U32|S64|U64].combiner
        # or ISETP.cmp.combiner (no size modifier)
        isUnsigned = False
        remaining = Inst.opcodes[2:]
        # Skip size modifiers (S32, U32, S64, U64) to find the combiner (AND/OR/XOR)
        cmp2_idx = 2
        for i, mod in enumerate(remaining):
            if mod in ("U32", "U64"):
                isUnsigned = True
                cmp2_idx = 2 + i + 1
            elif mod in ("S32", "S64"):
                isUnsigned = False
                cmp2_idx = 2 + i + 1
            elif mod in ("AND", "OR", "XOR"):
                cmp2_idx = 2 + i
                break
        cmp2 = Inst.opcodes[cmp2_idx] if cmp2_idx < len(Inst.opcodes) else "AND"

        is_extended = "EX" in Inst.opcodes

        funcs_map = {
            "EQ": "==",
            "NE": "!=",
            "LT": "<",
            "LE": "<=",
            "GT": ">",
            "GE": ">=",
        }

        cmp2_map = {
            "AND": IRBuilder.and_,
            "OR": IRBuilder.or_,
            "XOR": IRBuilder.xor,
        }

        if isUnsigned:
            r = IRBuilder.icmp_unsigned(funcs_map[cmp1], val1, val2, "isetrp_cmp1")
        else:
            r = IRBuilder.icmp_signed(funcs_map[cmp1], val1, val2, "isetrp_cmp1")

        i1 = self.ir.IntType(1)
        if is_extended:
            # ISETP.cmp.EX — extended 64-bit comparison.
            # uses[3] is the predicate from the low-word ISETP.
            # Semantics: if hi > hi2 → true; if hi < hi2 → false;
            #            if hi == hi2 → use low predicate.
            p_low = _get_val(Inst.get_uses()[3], "isetrp_ex_lo")
            # p_low may have been fetched as a wider integer (i32) when the
            # source pseudo-register was typed earlier; normalize to i1 so
            # the select branches share lhs/rhs types (both must equal r's i1).
            p_low = self._normalize_i1_in_curr_bb(IRBuilder, p_low, "isetrp_ex_lo")
            hi_eq = IRBuilder.icmp_unsigned("==", val1, val2, "isetrp_hi_eq")
            # When high words are equal, the comparison result comes from the
            # low-word predicate; otherwise, use the high-word comparison.
            r = IRBuilder.select(hi_eq, p_low, r, "isetrp_ex_sel")

        cmp2_func = cmp2_map.get(cmp2)
        if not cmp2_func:
            raise UnsupportedInstructionException

        # The 3rd use (Pc) is a predicate (i1) in SASS, but _get_val may have
        # fetched it as a wider integer when the source pseudo-register was
        # typed as i32 earlier (e.g., constant-fold of !PT). Normalize to i1
        # so IRBuilder.and_/or_/xor_ doesn't raise "Operands must be the same
        # type, got (i1, i32)". Observed on hecbench/frechet-cuda.
        # Additionally, when val3 is already i1 from a prior block whose def
        # does not dominate this use site (BUG-082), force the result through
        # an i32 carrier in the current block so the LLVM verifier sees a
        # local i1 def. This is harmless when SSA already placed a PHI here.
        val3 = self._normalize_i1_in_curr_bb(IRBuilder, val3, "isetrp_src")

        r1 = cmp2_func(r, val3, "isetrp_cmp2")

        if not dest1.is_pt:
            self._set_val(IRBuilder, IRRegs, dest1, r1)

        if not dest2.is_pt:
            r2 = IRBuilder.not_(r1, "isetrp_not")
            self._set_val(IRBuilder, IRRegs, dest2, r2)

    @lift_for("PBRA")
    def lift_pbra(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        pred = Inst.get_uses()[0]

        cond = _get_val(pred, "cond")
        # cond is a predicate (i1) in SASS, but _get_val may have widened it
        # to i32 when the source was typed as int (e.g., NOTYPE fallback or
        # cross-BB load through a wider SSA carrier). cbranch requires i1.
        cond = self._normalize_i1_in_curr_bb(IRBuilder, cond, "pbra_cond")

        true_br, false_br = Inst.parent.get_branch_pair(Inst, block_map.keys())
        IRBuilder.cbranch(cond, block_map[true_br], block_map[false_br])

    @lift_for("PACK64")
    def lift_pack64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        v1 = _get_val(uses[0])
        v2 = _get_val(uses[1])
        lo64 = IRBuilder.zext(v1, ir.IntType(64), "pack64_lo")
        hi64 = IRBuilder.zext(v2, ir.IntType(64), "pack64_hi")
        hiShift = IRBuilder.shl(hi64, ir.Constant(ir.IntType(64), 32), "pack64_hi_shift")
        packed = IRBuilder.or_(lo64, hiShift, "pack64_result")
        self._set_val(IRBuilder, IRRegs, dest, packed)

    @lift_for("UNPACK64")
    def lift_unpack64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dests = Inst.get_defs()
        src = Inst.get_uses()[0]
        val = _get_val(src)
        lo32 = IRBuilder.trunc(val, ir.IntType(32), "unpack64_lo")
        hi32_shift = IRBuilder.lshr(val, ir.Constant(ir.IntType(64), 32), "unpack64_hi_shift")
        hi32 = IRBuilder.trunc(hi32_shift, ir.IntType(32), "unpack64_hi")
        self._set_val(IRBuilder, IRRegs, dests[0], lo32)
        self._set_val(IRBuilder, IRRegs, dests[1], hi32)

    @lift_for("SGXT")
    def lift_sgxt(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        val = _get_val(Inst.get_uses()[0])
        sel = _get_val(Inst.get_uses()[1])
        
        # Extract one signed byte from a val by sel
        # then sign-extend it to 32-bit.
        idx = IRBuilder.and_(sel, ir.Constant(sel.type, 0x3), "sgxt_idx")
        shamt = IRBuilder.shl(idx, ir.Constant(idx.type, 3), "sgxt_shamt")
        temp = IRBuilder.lshr(val, shamt, "sgxt_shifted")
        temp = IRBuilder.trunc(temp, ir.IntType(8), "sgxt_trunc")
        res = IRBuilder.sext(temp, ir.IntType(32), "sgxt_sext")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("CAST64")
    def lift_cast64(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        op1 = Inst.get_uses()[0]
        if not dest.is_reg:
            raise UnsupportedInstructionException(f"CAST64 expects a register operand, got: {dest}")

        val = _get_val(op1)

        val64 = IRBuilder.sext(val, ir.IntType(64), "cast64")
        self._set_val(IRBuilder, IRRegs, dest, val64)

    @lift_for("BITCAST")
    def lift_bitcast(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        dest_type = dest.get_ir_type(self)

        val = _get_val(src)

        cast_val = IRBuilder.bitcast(val, dest_type, "cast")
        self._set_val(IRBuilder, IRRegs, dest, cast_val)

    @lift_for("POPC", "UPOPC")
    def lift_popc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        popcVal = self.run_intrinsic("ctpop_i32", IRBuilder, [val], "popc")
        self._set_val(IRBuilder, IRRegs, dest, popcVal)

    # NOTE: RED has no base-class handler — both backends (CpuLifter,
    # NVVMLifter) implement their own @lift_for("RED"); a former base
    # version with fuzzing-grade INC/DEC approximation was unreachable
    # (subclass entries win in _build_lift_map) and was removed 2026-07.

    def _f32_to_fp8_byte(self, IRBuilder, vf32, fmt, name):
        """f32 -> one fp8 byte (E4M3 or E5M2), RN-to-even, SATFINITE.

        Returns an i32 whose low 8 bits hold the fp8 encoding (upper bits zero).
        This is the bit-exact inverse of the F2FP.*.UNPACK_B decode and matches
        PTX `cvt.rn.satfinite.{e4m3,e5m2}x2.f32` (== CUDA's software
        `__nv_cvt_float_to_fp8`).  The previous PACK_AB code took the LOW BYTE
        of an f16 bit pattern, which is NOT an fp8 value (e.g. half(-2.0)=0xC000
        -> 0x00), zeroing most fp8 GEMM/FMHA outputs.

        fmt: "E4M3" (1.4.3, bias 7, max finite 448, S.1111.111=NaN, no Inf)
             "E5M2" (1.5.2, bias 15, max finite 57344, Inf/NaN like f16>>8).

        Method: operate on the f32 bit pattern.  Rebias the exponent, then round
        the (implicit-1 . 23-bit) significand down to the fp8 mantissa width with
        round-half-to-even, allowing the rounding carry to ripple into the
        exponent.  Subnormal inputs (target exp <= 0) shift the significand right
        by the underflow amount before rounding.  Overflow and NaN/Inf saturate
        to the format's max finite / NaN code.
        """
        i32 = self.ir.IntType(32)
        f32 = self.ir.FloatType()
        if vf32.type != f32:
            vf32 = self._insert_type_cast_if_needed(IRBuilder, vf32, f32, f"{name}_f32")

        if fmt == "E5M2":
            MBITS, BIAS = 2, 15
            MAXFIN = 0x7B            # S.11110.11 = 57344
            NANCODE = 0x7F
            INFCODE = 0x7C           # S.11111.00
            HAS_INF = True
        else:  # E4M3
            MBITS, BIAS = 3, 7
            MAXFIN = 0x7E            # S.1111.110 = 448
            NANCODE = 0x7F           # S.1111.111
            INFCODE = None
            HAS_INF = False

        C = lambda v: self.ir.Constant(i32, v)
        b = IRBuilder

        u = b.bitcast(vf32, i32, f"{name}_u")
        sign = b.and_(b.lshr(u, C(31), f"{name}_s0"), C(1), f"{name}_sign")
        sign8 = b.shl(sign, C(7), f"{name}_sign8")
        au = b.and_(u, C(0x7FFFFFFF), f"{name}_au")           # |x| f32 bits
        fexp = b.ashr(b.and_(au, C(0x7F800000), f"{name}_fe0"), C(23), f"{name}_fexp")
        fman = b.and_(au, C(0x7FFFFF), f"{name}_fman")        # 23-bit fraction

        # Significand with implicit leading 1 in bit 23: 1.fff... (24 bits).
        sig = b.or_(fman, C(1 << 23), f"{name}_sig")

        # Target unbiased exponent for a normal fp8: E = fexp - 127 + BIAS.
        E = b.add(b.sub(fexp, C(127), f"{name}_E0"), C(BIAS), f"{name}_E")

        # Right-shift: drop (23-MBITS) for normals, plus (1-E) more for
        # subnormals (E<=0).  shift>=24 rounds the value to zero.
        DROP = 23 - MBITS
        is_sub = b.icmp_signed("<=", E, C(0), f"{name}_issub")
        extra = b.sub(C(1), E, f"{name}_extra")               # 1-E (>0 when sub)
        extra = b.select(is_sub, extra, C(0), f"{name}_extra_sel")
        shift = b.add(C(DROP), extra, f"{name}_shift")
        too_small = b.icmp_signed(">", shift, C(31), f"{name}_toosmall")
        shift_c = b.select(too_small, C(31), shift, f"{name}_shift_c")

        # Round half-to-even on the dropped bits.
        kept = b.lshr(sig, shift_c, f"{name}_kept")
        one = C(1)
        shmask = b.sub(b.shl(one, shift_c, f"{name}_shm0"), one, f"{name}_shmask")
        rem = b.and_(sig, shmask, f"{name}_rem")
        halfbit = b.shl(one, b.sub(shift_c, one, f"{name}_h0"), f"{name}_half")
        gt_half = b.icmp_unsigned(">", rem, halfbit, f"{name}_gthalf")
        eq_half = b.icmp_unsigned("==", rem, halfbit, f"{name}_eqhalf")
        kept_odd = b.trunc(b.and_(kept, one, f"{name}_lsb0"), self.ir.IntType(1), f"{name}_odd")
        round_up = b.or_(gt_half, b.and_(eq_half, kept_odd, f"{name}_tieup"), f"{name}_rup")
        rounded = b.add(kept, b.select(round_up, one, C(0), f"{name}_rinc"), f"{name}_rounded")

        # `rounded` holds (for a normal) the magnitude with the implicit-1 bit
        # at position MBITS (== 1<<MBITS for E=1; a rounding carry promotes it).
        # Add (E-1)<<MBITS so the implicit bit becomes the low exponent bit.
        # Subnormals (E<=0) contribute 0; a carry promotes them to the smallest
        # normal automatically.
        Em1 = b.select(is_sub, C(0), b.sub(E, C(1), f"{name}_Em1s"), f"{name}_Em1")
        em = b.add(rounded, b.shl(Em1, C(MBITS), f"{name}_expc"), f"{name}_em")

        # Saturation / specials.
        is_inf_nan = b.icmp_unsigned("==", fexp, C(0xFF), f"{name}_finf")
        is_nan = b.fcmp_unordered("uno", vf32, vf32, f"{name}_isnan")
        overflow = b.icmp_unsigned(">", em, C(MAXFIN), f"{name}_ovf")
        if HAS_INF:
            spec = b.select(is_nan, C(NANCODE),
                            b.select(is_inf_nan, C(INFCODE), C(MAXFIN),
                                     f"{name}_infsel"),
                            f"{name}_nansel")
        else:
            spec = b.select(is_nan, C(NANCODE), C(MAXFIN), f"{name}_nansel")
        use_spec = b.or_(is_inf_nan, overflow, f"{name}_usespec")
        mag = b.select(use_spec, spec, em, f"{name}_mag")
        return b.or_(mag, sign8, f"{name}_fp8")

    def _to_i32_for_half2(self, IRBuilder, val, name):
        """Convert a value to i32 for half2 bitcast. Handles float immediates."""
        i32 = self.ir.IntType(32)
        if val.type == i32:
            return val
        if isinstance(val.type, (self.ir.FloatType,)):
            return IRBuilder.bitcast(val, i32, f"{name}_f2i")
        if isinstance(val.type, (self.ir.DoubleType,)):
            val = IRBuilder.fptrunc(val, self.ir.FloatType(), f"{name}_dbl2f")
            return IRBuilder.bitcast(val, i32, f"{name}_f2i")
        return self._insert_type_cast_if_needed(IRBuilder, val, i32, name)

    @lift_for("HADD2")
    def lift_hadd2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src1, src2 = Inst.get_uses()[0], Inst.get_uses()[1]

        is_bf16 = "BF16_V2" in Inst.opcodes
        i32 = self.ir.IntType(32)

        val1 = self._get_val_half2(IRBuilder, _get_val, src1, "hadd2_lhs", is_bf16=is_bf16)
        val2 = self._get_val_half2(IRBuilder, _get_val, src2, "hadd2_rhs", is_bf16=is_bf16)
        result = IRBuilder.fadd(val1, val2, "hadd2")

        # HADD2.F32: convert low half to float32 result (not valid for BF16_V2)
        if "F32" in Inst.opcodes:
            lo = IRBuilder.extract_element(result, self.ir.Constant(i32, 0), "hadd2_lo")
            result = IRBuilder.fpext(lo, self.ir.FloatType(), "hadd2_f32")
            result = IRBuilder.bitcast(result, i32, "hadd2_f32_i32")
        else:
            result = IRBuilder.bitcast(result, i32, "hadd2_i32")
        self._set_val(IRBuilder, IRRegs, dest, result)

    def _to_i32(self, IRBuilder, val, name):
        i32 = self.ir.IntType(32)
        if val.type == i32:
            return val
        if isinstance(val, self.ir.Constant) and isinstance(val.type, self.ir.FloatType):
            import struct
            int_bits = struct.unpack('<I', struct.pack('<f', float(val.constant)))[0]
            return self.ir.Constant(i32, int_bits)
        return self._insert_type_cast_if_needed(IRBuilder, val, i32, name)

    @staticmethod
    def _float_to_half_bits(f):
        """Convert a Python float to IEEE 754 half-precision (16-bit) integer."""
        import struct
        bits = struct.unpack('<I', struct.pack('<f', float(f)))[0]
        sign = (bits >> 16) & 0x8000
        exp = ((bits >> 23) & 0xFF) - 127 + 15
        frac = (bits >> 13) & 0x3FF
        if exp <= 0:
            return sign  # flush to zero for subnormals
        if exp >= 31:
            return sign | 0x7C00  # infinity
        return sign | (exp << 10) | frac

    @staticmethod
    def _float_to_half_bits_ieee(f):
        """Convert f32 -> f16 IEEE-754 bit pattern, preserving subnormals.

        Unlike `_float_to_half_bits` (which flushes subnormals to zero),
        this does proper IEEE-754 round-to-nearest-even and preserves the
        subnormal range. Required for the HFMA2.MMA constant-load idiom
        (BUG-088), where nvcc encodes integer constants whose low half
        often falls in the f16 subnormal range (e.g. sizeof(float)=4 is
        encoded as the f16 bit pattern 0x0004 = 4 * 2^-24, a subnormal).
        """
        try:
            import numpy as np
            return int(np.float16(float(f)).view(np.uint16))
        except Exception:
            # Pure-Python fallback (round-to-nearest-even).
            import struct
            u32 = struct.unpack('<I', struct.pack('<f', float(f)))[0]
            sign = (u32 >> 16) & 0x8000
            exp = ((u32 >> 23) & 0xff) - 127 + 15
            mant = u32 & 0x7fffff
            if exp <= 0:
                if exp < -10:
                    return sign
                mant |= 0x800000
                shift = 14 - exp
                r = mant >> shift
                if (mant >> (shift - 1)) & 1:
                    r += 1
                return sign | r
            elif exp >= 31:
                return sign | 0x7c00 | (0x200 if mant else 0)
            out = sign | (exp << 10) | (mant >> 13)
            if (mant & 0x1000) and ((mant & 0xfff) or (out & 1)):
                out += 1
            return out

    def _half2_elem_ty(self, is_bf16):
        """Return the element type for packed 16-bit float arithmetic.

        is_bf16=False => HalfType (IEEE 754 fp16)
        is_bf16=True  => BFloat16Type (brain float, emitted as `bfloat` in IR)
        """
        if is_bf16:
            from sir.bf16_type import BFloat16Type
            return BFloat16Type()
        return self.ir.HalfType()

    def _val_to_v2h(self, IRBuilder, val, name, is_bf16=False):
        """Bitcast an i32 (or equivalent) into a <2 x half> or <2 x bfloat>.

        Replaces the old _val_to_v2f16 which hardcoded half. Both packed
        layouts are 32-bit wide, so the bitcast is always legal.
        """
        import struct
        elem_ty = self._half2_elem_ty(is_bf16)
        v2 = self.ir.VectorType(elem_ty, 2)
        if val.type == v2:
            return val
        i32 = self.ir.IntType(32)
        # NOTE: the SASS-immediate case (e.g. the "1, 1" in
        # `HFMA2 R5, R5, R0.H0_H0, 1, 1`) is handled by the caller
        # `_get_val_half2` *before* we get here — `op.is_immediate` packs
        # f16(imm) into both lanes there. By the time `_val_to_v2h` runs,
        # `val` is the resolved register value, which is **always a bit
        # pattern**, not a float. The previous "small int constant ==
        # float immediate" path here re-interpreted the i32 holding
        # HFMA2.MMA's constant-load idiom (e.g. 0x3800 = packed
        # {0.5, 0.0}) as the *float* 14336.0 and re-packed it as
        # 0x73007300 — wrong by orders of magnitude. Bitcast i32 → v2
        # directly.
        if isinstance(val.type, (self.ir.FloatType,)):
            return IRBuilder.bitcast(val, v2, name)
        if val.type == i32:
            return IRBuilder.bitcast(val, v2, name)
        val = self._insert_type_cast_if_needed(IRBuilder, val, i32, name)
        return IRBuilder.bitcast(val, v2, name)

    def _val_to_v2f16(self, IRBuilder, val, name):
        """Backwards-compat wrapper — always packs as <2 x half>."""
        return self._val_to_v2h(IRBuilder, val, name, is_bf16=False)

    @staticmethod
    def _float_to_bf16_bits(f):
        """Convert a Python float to IEEE 754 bfloat16 (16-bit) integer.

        bfloat16 is the top 16 bits of a float32: same exponent, truncated
        mantissa. Round-to-nearest-even would be more accurate; this uses
        truncation which is acceptable for SASS-literal expansion.
        """
        import struct
        bits32 = struct.unpack('<I', struct.pack('<f', float(f)))[0]
        return (bits32 >> 16) & 0xFFFF

    def _get_val_half2(self, IRBuilder, _get_val, op, name, is_bf16=False):
        """Get a value for packed-half context. If the operand is an immediate,
        convert it as a 16-bit float (F16 or BF16) replicated to both lanes,
        rather than going through _get_val which corrupts float→i32.
        Handles .H0_H0 / .H1_H1 swizzle suffixes.

        When is_bf16=True, uses `<2 x bfloat>` and the bfloat bit layout.
        """
        elem_ty = self._half2_elem_ty(is_bf16)
        v2 = self.ir.VectorType(elem_ty, 2)
        i32 = self.ir.IntType(32)
        if op.is_immediate:
            fval = float(op.immediate_value)
            if is_bf16:
                h_bits = self._float_to_bf16_bits(fval)
            else:
                h_bits = self._float_to_half_bits(fval)
            packed = h_bits | (h_bits << 16)
            return IRBuilder.bitcast(self.ir.Constant(i32, packed), v2, name)

        # A packed-half2 operand's `-Rx` / `|Rx|` modifiers must be applied
        # PER-LANE on the <2 x half> vector (sign-bit flip / clear), NOT as i32
        # two's-complement on the packed 32-bit word. The generic _get_val would
        # integer-negate the whole word (e.g. -0x38003800 -> 0xC7FFC800, giving
        # h0 = -8.0 instead of -0.5), corrupting every half2 op with a negated
        # operand (e.g. the exp(-x) arg + range-reduction compares in the
        # half-precision SiLU/sigmoid). Strip the prefix so _get_val returns the
        # RAW packed bits, then redo neg/abs per-lane on v2.
        prefix = getattr(op, 'prefix', None)
        is_neg = bool(prefix and "-" in prefix)
        is_abs = bool(prefix and "|" in prefix)
        if is_neg or is_abs:
            saved_prefix = op.prefix
            op.prefix = (prefix or "").replace("-", "").replace("|", "") or None
            try:
                val = _get_val(op, name)
            finally:
                op.prefix = saved_prefix
        else:
            val = _get_val(op, name)
        val = self._val_to_v2h(IRBuilder, val, name, is_bf16=is_bf16)

        # Per-lane abs then negate (SASS operand modifier order: |.| then -).
        if is_abs:
            val = self._v2_fabs(IRBuilder, val, v2, f"{name}_v2abs")
        if is_neg:
            val = IRBuilder.fneg(val, f"{name}_v2neg")

        # Handle .H0_H0 / .H1_H1 swizzle: replicate one half to both lanes
        suffix = getattr(op, 'suffix', None) or ""
        if "H0_H0" in suffix:
            lo = IRBuilder.extract_element(val, self.ir.Constant(i32, 0), f"{name}_h0")
            val = IRBuilder.insert_element(val, lo, self.ir.Constant(i32, 1), f"{name}_h0h0")
        elif "H1_H1" in suffix:
            hi = IRBuilder.extract_element(val, self.ir.Constant(i32, 1), f"{name}_h1")
            val = IRBuilder.insert_element(val, hi, self.ir.Constant(i32, 0), f"{name}_h1h1")
        return val

    def _v2_fabs(self, IRBuilder, val, v2, name):
        """Per-lane fabs on a <2 x half>/<2 x bfloat>: clear each lane's sign
        bit (AND 0x7FFF per lane) via an i32 bit-op, then bitcast back."""
        i32 = self.ir.IntType(32)
        bits = IRBuilder.bitcast(val, i32, f"{name}_bits")
        masked = IRBuilder.and_(bits, self.ir.Constant(i32, 0x7FFF7FFF), f"{name}_mask")
        return IRBuilder.bitcast(masked, v2, f"{name}_v2")

    @lift_for("HFMA2")
    def lift_hfma2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        is_bf16 = "BF16_V2" in Inst.opcodes

        elem_ty = self._half2_elem_ty(is_bf16)
        v2 = self.ir.VectorType(elem_ty, 2)
        i32 = self.ir.IntType(32)

        is_mma = "MMA" in Inst.opcodes

        # BUG-088: nvcc constant-load idiom.
        # `HFMA2.MMA Rd, -RZ, RZ, imm_hi, imm_lo` (both multiply operands RZ)
        # is nvcc's one-instruction integer-constant-load trick: every f16
        # multiply contributes 0, so Rd's bit pattern is just the two f16
        # immediates concatenated. The downstream `IMAD.WIDE` reads Rd as a
        # 32-bit integer stride. We must emit the integer bit-pattern
        # directly (using IEEE-754 round-to-nearest-even with subnormal
        # preservation) instead of going through f16 math, because the f16
        # math path flushes subnormals (e.g. sizeof(float)=4 encodes as the
        # f16 bit pattern 0x0004 = subnormal -> would flush to 0).
        if (
            not is_bf16
            and is_mma
            and len(uses) >= 4
            and uses[0].is_rz
            and uses[1].is_rz
            and uses[2].is_immediate
            and uses[3].is_immediate
        ):
            c_hi = self._float_to_half_bits_ieee(float(uses[2].immediate_value))
            c_lo = self._float_to_half_bits_ieee(float(uses[3].immediate_value))
            lit_i32 = ((c_hi & 0xFFFF) << 16) | (c_lo & 0xFFFF)
            result = self.ir.Constant(i32, lit_i32)
            self._set_val(IRBuilder, IRRegs, dest, result)
            return

        # Choose immediate packer based on BF16 vs F16
        if is_bf16:
            pack_imm = self._float_to_bf16_bits
        else:
            pack_imm = self._float_to_half_bits

        if len(uses) >= 4 and uses[1].is_immediate and uses[2].is_immediate:
            # HFMA2[.MMA] Rd, Ra, imm_b_hi, imm_b_lo, Rc
            a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hfma2_a", is_bf16=is_bf16)
            b_hi = pack_imm(float(uses[1].immediate_value))
            b_lo = pack_imm(float(uses[2].immediate_value))
            b_packed = b_lo | (b_hi << 16)
            b = IRBuilder.bitcast(self.ir.Constant(i32, b_packed), v2, "hfma2_b")
            c = self._get_val_half2(IRBuilder, _get_val, uses[3], "hfma2_c", is_bf16=is_bf16)
        elif is_mma and len(uses) >= 4 and uses[1].is_immediate:
            # HFMA2.MMA Rd, Ra, imm_b, Rb_lo, Rc — mixed immediate+register
            a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hfma2_a", is_bf16=is_bf16)
            b = self._get_val_half2(IRBuilder, _get_val, uses[1], "hfma2_b", is_bf16=is_bf16)
            c = self._get_val_half2(IRBuilder, _get_val, uses[3] if len(uses) > 3 else uses[2],
                                    "hfma2_c", is_bf16=is_bf16)
        elif len(uses) >= 4 and uses[2].is_immediate and uses[3].is_immediate:
            # HFMA2[.MMA] Rd, Ra, Rb, imm_c_hi, imm_c_lo
            a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hfma2_a", is_bf16=is_bf16)
            b = self._get_val_half2(IRBuilder, _get_val, uses[1], "hfma2_b", is_bf16=is_bf16)
            c_hi = pack_imm(float(uses[2].immediate_value))
            c_lo = pack_imm(float(uses[3].immediate_value))
            c_packed = c_lo | (c_hi << 16)
            c = IRBuilder.bitcast(self.ir.Constant(i32, c_packed), v2, "hfma2_c")
        else:
            # Standard HFMA2 Rd, Ra, Rb, Rc (3 register uses)
            a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hfma2_a", is_bf16=is_bf16)
            b = self._get_val_half2(IRBuilder, _get_val, uses[1], "hfma2_b", is_bf16=is_bf16)
            c = self._get_val_half2(IRBuilder, _get_val, uses[2], "hfma2_c", is_bf16=is_bf16)

        prod = IRBuilder.fmul(a, b, "hfma2_mul")
        result_v = IRBuilder.fadd(prod, c, "hfma2_res")
        result = IRBuilder.bitcast(result_v, self.ir.IntType(32), "hfma2")
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("HMUL2")
    def lift_hmul2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        is_bf16 = "BF16_V2" in Inst.opcodes

        a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hmul2_a", is_bf16=is_bf16)
        b = self._get_val_half2(IRBuilder, _get_val, uses[1], "hmul2_b", is_bf16=is_bf16)
        result_v = IRBuilder.fmul(a, b, "hmul2_res")
        result = IRBuilder.bitcast(result_v, self.ir.IntType(32), "hmul2")
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("HMNMX2")
    def lift_hmnmx2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()

        # Route operands through _get_val_half2 so .H0_H0 / .H1_H1 swizzle is
        # honored (broadcast one half to both lanes) and any `-Rx` / `|Rx|`
        # modifier is applied PER-LANE. The generic _get_val + raw bitcast
        # would DROP the swizzle (e.g. `Rb.H0_H0` = broadcast s to both lanes
        # becomes {s, 0.0} from a 16-bit zext load) and integer-negate the
        # packed word — corrupting the high lane of the min/max. The .H0_H0
        # broadcast form appears 32x in CB_hgemm + cutlass 13_fused_two_gemms.
        # (No bf16 HMNMX2 form in the corpus — default <2 x half> is correct.)
        a = self._get_val_half2(IRBuilder, _get_val, uses[0], "hmnmx2_a")
        b = self._get_val_half2(IRBuilder, _get_val, uses[1], "hmnmx2_b")

        is_min = "MIN" in Inst.opcodes
        has_ftz = "FTZ" in Inst.opcodes
        if is_min:
            intrinsic_name = "hmin2_ftz" if has_ftz else "hmin2"
        else:
            intrinsic_name = "hmax2_ftz" if has_ftz else "hmax2"

        result_v = self.run_intrinsic(intrinsic_name, IRBuilder, [a, b], intrinsic_name)
        result = IRBuilder.bitcast(result_v, self.ir.IntType(32), "hmnmx2")
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("HSETP2")
    def lift_hsetp2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        dest1, dest2 = defs[0], defs[1]
        ra, rb = uses[0], uses[1]
        psrc = uses[2]

        # ra/rb are packed-half2: route through _get_val_half2 so their
        # `-Rx` / `|Rx|` modifiers are applied PER-LANE (sign-bit flip / clear)
        # and any .H0_H0 / .H1_H1 swizzle is honored. Using the generic _get_val
        # here would integer-negate the packed word and drop the swizzle,
        # corrupting the half-precision range-reduction predicates (e.g. in the
        # SiLU/sigmoid fast_exp path).
        v2f16 = self.ir.VectorType(self.ir.HalfType(), 2)
        a = self._get_val_half2(IRBuilder, _get_val, ra, "hsetp2_a")
        b = self._get_val_half2(IRBuilder, _get_val, rb, "hsetp2_b")
        val3 = _get_val(psrc)

        cmp_op = Inst.opcodes[1]
        bool_op_idx = 2
        if len(Inst.opcodes) > 2 and Inst.opcodes[2] in ("FTZ", "NAN"):
            bool_op_idx = 3
        bool_op = Inst.opcodes[bool_op_idx] if len(Inst.opcodes) > bool_op_idx else "AND"

        fcmp_map = {
            "EQ": "==", "NE": "!=", "LT": "<", "LE": "<=", "GT": ">", "GE": ">=",
            "NEU": "!=", "LTU": "<", "LEU": "<=", "GTU": ">", "GEU": ">="
        }
        is_unordered = cmp_op.endswith("U") and cmp_op != "NEU"
        fcmp_pred = fcmp_map.get(cmp_op, "==")

        if is_unordered or cmp_op in ("NEU", "LTU", "LEU", "GTU", "GEU"):
            cmp_result = IRBuilder.fcmp_unordered(fcmp_pred, a, b, "hsetp2_cmp")
        else:
            cmp_result = IRBuilder.fcmp_ordered(fcmp_pred, a, b, "hsetp2_cmp")

        pred0 = IRBuilder.extract_element(cmp_result, self.ir.Constant(self.ir.IntType(32), 0), "hsetp2_cmp0")
        pred1 = IRBuilder.extract_element(cmp_result, self.ir.Constant(self.ir.IntType(32), 1), "hsetp2_cmp1")

        bool_func_map = {"AND": IRBuilder.and_, "OR": IRBuilder.or_, "XOR": IRBuilder.xor}
        bool_func = bool_func_map.get(bool_op, IRBuilder.and_)
        if val3.type != self.ir.IntType(1):
            val3 = IRBuilder.trunc(val3, self.ir.IntType(1), "hsetp2_pred_trunc")
        r0 = bool_func(pred0, val3, "hsetp2_r0")
        r1 = bool_func(pred1, val3, "hsetp2_r1")

        if not dest1.is_pt:
            self._set_val(IRBuilder, IRRegs, dest1, r0)
        if not dest2.is_pt:
            self._set_val(IRBuilder, IRRegs, dest2, r1)

    @lift_for("HSET2")
    def lift_hset2(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        uses = Inst.get_uses()
        ra, rb = uses[0], uses[1]
        psrc = uses[2]

        # ra/rb are packed-half2: route through _get_val_half2 so their
        # `-Rx` / `|Rx|` modifiers are applied PER-LANE (sign-bit flip / clear)
        # and any .H0_H0 / .H1_H1 swizzle is honored. The generic _get_val +
        # raw bitcast would integer-negate the packed word and drop the swizzle
        # (e.g. -0x38003800 -> 0xC7FFC800 => lane0 = -8.0 instead of -0.5),
        # corrupting the half-precision compare mask — the same defect that was
        # fixed in lift_hsetp2. (`HSET2.BF` = boolean OUTPUT format, the inputs
        # are f16x2, so default <2 x half> is correct.)
        a = self._get_val_half2(IRBuilder, _get_val, ra, "hset2_a")
        b = self._get_val_half2(IRBuilder, _get_val, rb, "hset2_b")
        val3 = _get_val(psrc)

        i1 = self.ir.IntType(1)
        i32 = self.ir.IntType(32)
        # Normalize val3 to a clean i1. Two scenarios occur in practice:
        #   (a) val3.type is non-i1 (float / half / iN) — needs a
        #       type-cast + nonzero comparison.
        #   (b) val3 claims type==i1 but its inner Python value is a
        #       float (parser bug: an HSET2 source typed as predicate
        #       that actually holds a half-precision immediate).
        #       llvmlite emits invalid IR for `and i1 cmp, <float>` —
        #       llc rejects with "floating point constant invalid for
        #       type". Treat any nonzero float bit-pattern as true.
        if isinstance(val3, ir.Constant) and isinstance(val3.constant, float):
            val3 = self.ir.Constant(i1, 0 if val3.constant == 0.0 else 1)
        elif val3.type != i1:
            val3 = self._insert_type_cast_if_needed(IRBuilder, val3, i32, "hset2_pred_i32")
            val3 = IRBuilder.icmp_unsigned("!=", val3, self.ir.Constant(i32, 0), "hset2_pred")

        cmp_op = Inst.opcodes[2]
        bool_op_idx = 3
        if len(Inst.opcodes) > 3 and Inst.opcodes[3] in ("FTZ", "NAN"):
            bool_op_idx = 4
        bool_op = Inst.opcodes[bool_op_idx] if len(Inst.opcodes) > bool_op_idx else "AND"

        fcmp_map = {
            "EQ": "==", 
            "NE": "!=", 
            "LT": "<", 
            "LE": "<=", 
            "GT": ">", 
            "GE": ">=",
            "NEU": "!=",
            "LTU": "<",
            "LEU": "<=",
            "GTU": ">",
            "GEU": ">=",
        }
        is_unordered = cmp_op.endswith("U")
        fcmp_pred = fcmp_map.get(cmp_op, "==")

        if is_unordered:
            cmp_result = IRBuilder.fcmp_unordered(fcmp_pred, a, b, "hset2_cmp")
        else:
            cmp_result = IRBuilder.fcmp_ordered(fcmp_pred, a, b, "hset2_cmp")

        i16 = self.ir.IntType(16)
        i32 = self.ir.IntType(32)
        v2i16 = self.ir.VectorType(i16, 2)
        # BUG-033: HSET2.BF produces bitmask 0xFFFF/0x0000 on GPU hardware,
        # but GPU FP16 multiply treats 0xFFFF (NaN) as a bitwise AND mask.
        # On x86 with IEEE-compliant FP, 0xFFFF * x = NaN. Use half(1.0)
        # = 0x3C00 for true so downstream HFMA2/HMUL2 FP multiply works.
        mask_true = self.ir.Constant(i16, 0x3C00)  # half 1.0
        mask_false = self.ir.Constant(i16, 0)

        cmp0 = IRBuilder.extract_element(cmp_result, self.ir.Constant(i32, 0), "hset2_cmp0")
        cmp1 = IRBuilder.extract_element(cmp_result, self.ir.Constant(i32, 1), "hset2_cmp1")

        bool_func_map = {"AND": IRBuilder.and_, "OR": IRBuilder.or_, "XOR": IRBuilder.xor}
        bool_func = bool_func_map.get(bool_op, IRBuilder.and_)
        r0 = bool_func(cmp0, val3, "hset2_r0")
        r1 = bool_func(cmp1, val3, "hset2_r1")

        m0 = IRBuilder.select(r0, mask_true, mask_false, "hset2_m0")
        m1 = IRBuilder.select(r1, mask_true, mask_false, "hset2_m1")

        vec = IRBuilder.insert_element(self.ir.Constant(v2i16, [0, 0]), m0, self.ir.Constant(i32, 0), "hset2_v0")
        vec = IRBuilder.insert_element(vec, m1, self.ir.Constant(i32, 1), "hset2_v1")
        result = IRBuilder.bitcast(vec, i32, "hset2")
        self._set_val(IRBuilder, IRRegs, dest, result)
        
    @lift_for("BREAK")
    def lift_break(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # BREAK Pd = B<n>: tests the convergence barrier and sets Pd.
        # On x86 (single-threaded), the barrier is always satisfied so
        # BREAK never fires.  Set the predicate to false so the
        # subsequent BRA Pd falls through (loop continues).
        defs = Inst.get_defs()
        if defs and defs[0] and defs[0].is_writable_reg:
            false_val = self.ir.Constant(self.ir.IntType(1), 0)
            self._set_val(IRBuilder, IRRegs, defs[0], false_val)

    def _bar_operand_i32(self, IRBuilder, v):
        """Coerce a BAR id/count operand (immediate or register) to i32."""
        i32 = self.ir.IntType(32)
        if isinstance(v.type, self.ir.PointerType):
            return IRBuilder.ptrtoint(v, i32, "bar_op_p")
        if hasattr(v.type, "width"):
            if v.type.width < 32:
                return IRBuilder.zext(v, i32, "bar_op_z")
            if v.type.width > 32:
                return IRBuilder.trunc(v, i32, "bar_op_t")
        return v

    @lift_for("BAR")
    def lift_bar(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        if len(Inst.opcodes) < 2:
            self.run_intrinsic("barrier", IRBuilder, [], "barrier0")
            return
            
        sub_op = Inst.opcodes[1]
        
        if sub_op in ("SYNC", "ARV", "ALL", "LE", "SC"):
            # BAR.SYNC — block-level barrier (NOP on x86 single-thread)
            # BAR.ARV — barrier arrive (half of split-phase barrier)
            # BAR.ALL — barrier for all threads (with scope: .CTA, .GPU)
            # BAR.LE — barrier with less-than-or-equal count
            # BAR.SC — barrier with scope (.CTA, .GPU)
            # All are NOPs on x86 single-thread.
            # Keep the barrier intrinsic call for BAR.SYNC (existing behavior).
            if sub_op == "SYNC":
                # NAMED barrier `BAR.SYNC <id>, <count>` (>=2 operands: a barrier
                # id + a participant count, e.g. FA3's `BAR.SYNC 0x1, 0x100`) is a
                # PARTIAL/warpgroup barrier over `count` threads at named barrier
                # `id` -- NOT the full-CTA __syncthreads. Lifting it as the
                # full-CTA `barrier` deadlocks a warp-specialized kernel. Lower it
                # to a real PARTIAL named barrier: the wrapper synchronizes exactly
                # `count` participants at barrier `id` (a fiber yield on the warp
                # backend; a NOP on the per-thread x86 backend, which has no notion
                # of warpgroups, so that path stays byte-identical). Plain
                # __syncthreads is `BAR.SYNC[.DEFER] 0x0` (single id operand).
                uses = Inst.get_uses()
                if len(uses) >= 2:
                    nid = self._bar_operand_i32(IRBuilder, _get_val(uses[0], "nbar_id"))
                    ncnt = self._bar_operand_i32(IRBuilder, _get_val(uses[1], "nbar_cnt"))
                    self.run_intrinsic("named_barrier_sync", IRBuilder, [nid, ncnt], "nbar")
                    return
                self.run_intrinsic("barrier", IRBuilder, [], "barrier0")
            elif sub_op == "ARV" and len(Inst.get_uses()) >= 2:
                # BAR.ARV <id>, <count> -- split-phase named barrier ARRIVE (no
                # wait). Registers this thread's arrival at named barrier `id` so a
                # peer warpgroup's BAR.SYNC over the same `count` can complete.
                uses = Inst.get_uses()
                nid = self._bar_operand_i32(IRBuilder, _get_val(uses[0], "narv_id"))
                ncnt = self._bar_operand_i32(IRBuilder, _get_val(uses[1], "narv_cnt"))
                self.run_intrinsic("named_barrier_arrive", IRBuilder, [nid, ncnt], "nbar_arv")

        elif sub_op == "RED":
            red_op = Inst.opcodes[2] if len(Inst.opcodes) > 2 else None
            if red_op == "AND":
                pred_val = _get_val(Inst.get_uses()[1], "bar_red_pred")
                pred_i32 = IRBuilder.zext(pred_val, self.ir.IntType(32), "bar_red_pred_i32")
                IRRegs["__bar_red_result"] = self.run_intrinsic("barrier0_and", IRBuilder, [pred_i32], "bar_red_and")
            elif red_op == "OR":
                pred_val = _get_val(Inst.get_uses()[1], "bar_red_pred")
                pred_i32 = IRBuilder.zext(pred_val, self.ir.IntType(32), "bar_red_pred_i32")
                IRRegs["__bar_red_result"] = self.run_intrinsic("barrier0_or", IRBuilder, [pred_i32], "bar_red_or")
            elif red_op == "POPC":
                pred_i32 = self.ir.Constant(self.ir.IntType(32), 1)
                IRRegs["__bar_red_result"] = self.run_intrinsic("barrier0_popc", IRBuilder, [pred_i32], "bar_red_popc")
            else:
                raise UnsupportedInstructionException(f"Unknown BAR.RED variant: {red_op}")
        else:
            raise UnsupportedInstructionException(f"Unknown BAR variant: {sub_op}")

    @lift_for("B2R")
    def lift_b2r(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        if "RESULT" not in Inst.opcodes:
            raise UnsupportedInstructionException(f"Unknown B2R variant: {Inst.opcodes}")
        
        defs = Inst.get_defs()
        rdst, pdst = defs[0], defs[1]
        
        res = IRRegs.get("__bar_red_result")
        if res is None:
            res = self.ir.Constant(self.ir.IntType(32), 0)
        
        if not rdst.is_rz:
            self._set_val(IRBuilder, IRRegs, rdst, res)
        
        if not pdst.is_pt:
            res_bool = IRBuilder.icmp_unsigned("!=", res, self.ir.Constant(self.ir.IntType(32), 0), "b2r_pred")
            self._set_val(IRBuilder, IRRegs, pdst, res_bool)

    @lift_for("P2R")
    def lift_p2r(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """P2R Rd, PR, Rs, mask — Pack predicate registers into integer register.

        Reads predicate registers P0-P6 (7 predicates), packs their boolean values
        into the selected byte in Rs and stores the combined value to Rd.

        P2R R20, PR, RZ, 0x20  → R20 = (P5 << 5)  (mask 0x20 = bit 5 = P5)
        P2R R0, PR, RZ, 0x1   → R0 = P0            (mask 0x01 = bit 0 = P0)
        P2R R0, PR, RZ, 0x7f  → R0 = P6<<6 | P5<<5 | ... | P0  (all 7 predicates)
        """
        defs = Inst.get_defs()
        uses = Inst.get_uses()
        dest = defs[0]

        # Rs = base value to OR into (usually RZ = 0)
        rs_op = uses[1] if len(uses) > 1 else None
        if rs_op and not rs_op.is_rz:
            base_val = _get_val(rs_op, "p2r_base")
            base_val = self._insert_type_cast_if_needed(
                IRBuilder, base_val, self.ir.IntType(32), "p2r_base_i32"
            )
        else:
            base_val = self.ir.Constant(self.ir.IntType(32), 0)

        # mask = which predicate bits to read
        mask_op = uses[2] if len(uses) > 2 else uses[-1]
        if mask_op.is_immediate:
            mask = mask_op.immediate_value
        else:
            mask = int(mask_op.name, 0) if mask_op.name.startswith("0x") else int(mask_op.name)

        i32 = self.ir.IntType(32)
        i1 = self.ir.IntType(1)
        byte_off = next((8 * int(mod[1]) for mod in Inst.opcodes[1:] if mod in ("B0", "B1", "B2", "B3")), 0)
        keep_mask = 0xffffffff ^ ((mask & 0x7f) << byte_off)
        result = IRBuilder.and_(base_val, self.ir.Constant(i32, keep_mask), "p2r_base_masked")

        # The masked physical predicates are explicit use operands (appended by
        # Instruction.__init__ in ascending bit order, after [PR, Rs, mask]), so
        # SSA/def-use give us each one's correct reaching value — cross-BB and
        # loop-carried included, with PHIs where needed. OR each into its bit
        # position. A predicate with no reaching def reads as 0 via _get_val's
        # SETZERO handling.
        appended_preds = list(uses[3:])  # uses = [PR, Rs, mask, <pred ops...>]
        ai = 0
        for bit in range(7):  # P0-P6
            if mask & (1 << bit):
                pred_op = appended_preds[ai] if ai < len(appended_preds) else None
                ai += 1
                if pred_op is not None and not (pred_op.is_pt and not pred_op.is_not_reg):
                    pred_val = _get_val(pred_op, f"p2r_p{bit}")
                else:
                    pred_val = self.ir.Constant(i1, 0)
                if hasattr(pred_val.type, 'width') and pred_val.type.width != 1:
                    pred_val = IRBuilder.trunc(pred_val, i1, f"p2r_p{bit}_i1")
                pred_i32 = IRBuilder.zext(pred_val, i32, f"p2r_p{bit}_i32")
                shifted = IRBuilder.shl(
                    pred_i32, self.ir.Constant(i32, bit + byte_off), f"p2r_p{bit}_shl"
                )
                result = IRBuilder.or_(result, shifted, f"p2r_or_{bit}")

        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("R2P")
    def lift_r2p(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        """R2P PR, Rs, mask — Unpack integer register bits into predicate registers.

        The masked predicates are explicit DEF operands (Instruction.__init__,
        ascending bit order) so SSA versions them and downstream uses link to
        these defs. Writes Rs[bit] into each.

        R2P PR, R0, 0x3   → P0 = R0[0], P1 = R0[1]          (mask 0x03)
        R2P PR, R0, 0x7f  → P0..P6 = R0[0..6]                (mask 0x7f)
        """
        defs = Inst.get_defs()   # the predicate operands (ascending bit order)
        uses = Inst.get_uses()   # [Rs, mask]
        rs_op = uses[0]
        src_val = _get_val(rs_op, "r2p_src")
        src_val = self._insert_type_cast_if_needed(
            IRBuilder, src_val, self.ir.IntType(32), "r2p_src_i32"
        )

        mask_op = uses[1] if len(uses) > 1 else uses[-1]
        if mask_op.is_immediate:
            mask = mask_op.immediate_value
        else:
            mask = int(mask_op.name, 0) if mask_op.name.startswith("0x") else int(mask_op.name)

        i32 = self.ir.IntType(32)
        i1 = self.ir.IntType(1)

        # Honor the byte selector (R0.B1/.B2/.B3): R2P unpacks bits from the
        # SELECTED byte of the source, not always byte 0. randomAccess's
        # HPCC_starts GF(2) matrix-power jump-ahead tests bits 8-30 via
        # .B1/.B2/.B3 — without the offset they all read byte 0 and every
        # thread's RNG stream (hence the whole table) is wrong (qrg too).
        byte_off = 0
        suf = getattr(rs_op, "suffix", None)
        if suf and len(suf) == 2 and suf[0] in "Bb" and suf[1].isdigit():
            byte_off = int(suf[1]) * 8

        di = 0
        for bit in range(7):  # P0-P6
            if mask & (1 << bit):
                shifted = IRBuilder.lshr(
                    src_val, self.ir.Constant(i32, bit + byte_off), f"r2p_shr_{bit}"
                )
                bit_val = IRBuilder.trunc(shifted, i1, f"r2p_p{bit}")
                if di < len(defs):
                    self._set_val(IRBuilder, IRRegs, defs[di], bit_val)
                    di += 1

    @lift_for("BREV", "UBREV")
    def lift_brev(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src, "brev_val")

        width = val.type.width
        intrinsic_name = "brev64" if width == 64 else "brev32"
        revVal = self.run_intrinsic(intrinsic_name, IRBuilder, [val], intrinsic_name)
        self._set_val(IRBuilder, IRRegs, dest, revVal)

    @lift_for("FLO", "UFLO")
    def lift_flo(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        type_tok = Inst.opcodes[1]
        mode = Inst.opcodes[2] if len(Inst.opcodes) > 2 else None

        type_map = {"U64": ("ctlz_i64", 63), "U32": ("ctlz_i32", 31)}
        entry = type_map.get(type_tok)
        if entry is None:
            raise UnsupportedInstructionException(f"Unsupported FLO type {type_tok}")
        intrinsic_name, msb = entry

        if mode is not None and mode != "SH":
            raise UnsupportedInstructionException(f"Unsupported FLO mode {mode}")

        clzVal = self.run_intrinsic(intrinsic_name, IRBuilder, [val], "flo_clz")
        if mode == "SH":
            # FLO.<type>.SH is the normalizing shift amount: clz(x) for x != 0,
            # 0xFFFFFFFF for x == 0 (llvm.ctlz alone yields the bit width).
            zero = IRBuilder.icmp_unsigned(
                "==", val, self.ir.Constant(val.type, 0), "flo_zero")
            floVal = IRBuilder.select(
                zero, self.ir.Constant(clzVal.type, -1), clzVal, "flo")
        else:
            # Plain FLO returns the BIT INDEX of the most-significant set bit
            # (31 - clz for .U32; -1 = 0xFFFFFFFF for x == 0, which 31 - 32
            # yields naturally). It is NOT clz: cub's match-based radix rank
            # computes the warp leader as FLO(peer_mask) and compares it to
            # the lane id — clz semantics make the leader test never fire, so
            # the leader-only ATOMS.ADD is skipped and every rank collapses
            # to popc-1 (sa-cuda / sortKV-cuda Onesweep corruption).
            floVal = IRBuilder.sub(
                self.ir.Constant(clzVal.type, msb), clzVal, "flo")
        self._set_val(IRBuilder, IRRegs, dest, floVal)

    def _collective_site(self, Inst):
        """Stable per-static-collective site id: the SASS instruction address
        (same convention as the BSSY/BSYNC history events in lift_ssy)."""
        return self.ir.Constant(self.ir.IntType(32), int(Inst.id.split('_')[0], 16))

    def _shfl_membermask(self, Inst, mode, IRBuilder, IRRegs, _get_val):
        """Membermask for the `shfl.sync` lowering.

        Base/x86/cpu: the membermask is a runtime concept (the warp wrapper
        gathers lanes from a fixed-size group), so always return the all-ones
        constant -1 — backend-safe and matching the prior behavior.

        Overridden on the NVVM backend to recover a partial-warp active mask
        for the warp-aggregated-atomic idiom (see LifterNVVM._shfl_membermask).
        """
        return self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)

    @lift_for("SHFL")
    def lift_shfl(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        destPred = Inst.get_defs()[0]
        destReg = Inst.get_defs()[1]
        srcReg = Inst.get_uses()[0]
        offset = Inst.get_uses()[1]
        width = Inst.get_uses()[2]

        val = _get_val(srcReg, "shfl_val")
        off = _get_val(offset, "shfl_offset")
        wid = _get_val(width, "shfl_width")

        mode = Inst.opcodes[1].lower()
        dtype = destReg.get_ir_type(self)
        mask_val = self._shfl_membermask(Inst, mode, IRBuilder, IRRegs, _get_val)

        intrinsic_name = f"shfl_{mode}_{dtype.intrinsic_name}"
        shfl_args = [mask_val, val, off, wid]
        if self.collective_site_arg:
            shfl_args.append(self._collective_site(Inst))
        shflVal = self.run_intrinsic(intrinsic_name, IRBuilder, shfl_args, f"shfl_{mode}")

        self._set_val(IRBuilder, IRRegs, destReg, shflVal)
        if not destPred.is_pt:
            # The destination predicate is PTX `pval`: whether the computed
            # source lane falls inside the shuffle segment. cub's warp scan
            # guards its accumulate step with it (SHFL.UP P0 ... @P0 IADD3) —
            # a constant-TRUE pred makes lane 0 add its own value twice and
            # corrupts every BlockScan prefix. Compute it per the PTX spec:
            #   cval = c[4:0]; segmask = c[12:8]
            #   minLane = lane & segmask
            #   maxLane = minLane | (cval & ~segmask)
            #   UP:   j = lane - b[4:0]; pval = j >= minLane
            #   DOWN: j = lane + b[4:0]; pval = j <= maxLane
            #   BFLY: j = lane ^ b[4:0]; pval = j <= maxLane
            #   IDX:  j = minLane | (b[4:0] & ~segmask); pval = j <= maxLane
            # (matches the runtime wrapper's own-value fallback window).
            i32 = self.ir.IntType(32)
            lane = self._emit_lane_id_i32(IRBuilder)
            b32 = self._insert_type_cast_if_needed(IRBuilder, off, i32, "shfl_b")
            c32 = self._insert_type_cast_if_needed(IRBuilder, wid, i32, "shfl_c")
            c31 = self.ir.Constant(i32, 31)
            b5 = IRBuilder.and_(b32, c31, "shfl_b5")
            cval = IRBuilder.and_(c32, c31, "shfl_cval")
            segmask = IRBuilder.and_(
                IRBuilder.lshr(c32, self.ir.Constant(i32, 8)), c31, "shfl_segmask")
            minLane = IRBuilder.and_(lane, segmask, "shfl_minlane")
            notseg = IRBuilder.xor(segmask, self.ir.Constant(i32, -1), "shfl_notseg")
            maxLane = IRBuilder.or_(
                minLane, IRBuilder.and_(cval, notseg), "shfl_maxlane")
            if mode == "up":
                j = IRBuilder.sub(lane, b5, "shfl_j")
                pval = IRBuilder.icmp_signed(">=", j, minLane, "shfl_pval")
            elif mode == "down":
                j = IRBuilder.add(lane, b5, "shfl_j")
                pval = IRBuilder.icmp_signed("<=", j, maxLane, "shfl_pval")
            elif mode == "bfly":
                j = IRBuilder.xor(lane, b5, "shfl_j")
                pval = IRBuilder.icmp_signed("<=", j, maxLane, "shfl_pval")
            elif mode == "idx":
                j = IRBuilder.or_(
                    minLane, IRBuilder.and_(b5, notseg), "shfl_j")
                pval = IRBuilder.icmp_signed("<=", j, maxLane, "shfl_pval")
            else:
                raise UnsupportedInstructionException(
                    f"Unsupported SHFL mode {mode} for destination predicate")
            self._set_val(IRBuilder, IRRegs, destPred, pval)

    @staticmethod
    def _prmt_a_is_cluster_rank(Inst, a_op):
        """True iff PRMT operand a is (transitively) the CGA CTA rank — an
        S2UR/S2R read of SR_CgaCtaId — walking reaching_defs through pure
        value-moving relays (MOV/UMOV/PHI, bounded depth).

        Used to gate the mapa-base collapse in lift_prmt: selector 0x654 alone
        does not prove the mapa pattern (a genuine data byte-permute could use
        the same selector in a UBLKCP kernel), so additionally require that the
        high-byte source really is the cluster rank. ptxas materializes the
        rank with a fresh S2UR near each mapa site (verified across the FA3
        corpus), so the chain is short; PHI is accepted as a relay in case SSA
        placed one across a predicated-region split."""
        seen = set()
        frontier = [(Inst, a_op, 0)]
        while frontier:
            cur_inst, op, depth = frontier.pop()
            if depth > 8:
                continue
            rd = getattr(cur_inst, "reaching_defs", None)
            def_inst = rd.get(op) if rd else None
            if def_inst is None or def_inst is cur_inst:
                continue
            if id(def_inst) in seen:
                continue
            seen.add(id(def_inst))
            opc0 = def_inst.opcodes[0] if def_inst.opcodes else ""
            if opc0 in ("S2UR", "S2R"):
                if any(getattr(u, "is_cluster_ctarank", False)
                       for u in def_inst.get_uses()):
                    return True
                continue
            if opc0 in ("MOV", "UMOV", "PHI"):
                for u in def_inst.get_uses():
                    if getattr(u, "is_reg", False):
                        frontier.append((def_inst, u, depth + 1))
        return False

    @lift_for("PRMT", "UPRMT")
    def lift_prmt(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        a_op, sel_op, b_op = Inst.get_uses()[0], Inst.get_uses()[1], Inst.get_uses()[2]
        i32 = self.ir.IntType(32)
        i64 = self.ir.IntType(64)
        i8 = self.ir.IntType(8)

        # Hopper cluster-shared `mapa` lowering: `PRMT d, rank, 0x654, window_base`
        # builds a distributed-shared (DSMEM) base = (cta_rank<<24) | window_base
        # (selector 0x654 puts operand-a's byte0 = the CTA-rank in the high byte and
        # operand-b's low 3 bytes = the DSMEM window base in the low). On the x86
        # single-CTA (cluster-of-1) backend a DSMEM self-address aliases the local
        # shared window, so this base MUST be 0 — otherwise cluster-addressed ops
        # (UBLKCP/UBLKRED/mbarriers) land at local+window_base while local LDS/STS
        # read local+0, and they miss each other (the FA3 bwd LSE tile is bulk-copied
        # to local+0x400 but read locally -> reads zero -> P=1 -> dV/dK/dQ garbage).
        # Gated on _cluster_is_local (x86 only; nvvm keeps the real cluster map),
        # _has_ublkcp (only cluster-bulk-copy kernels mix cluster-write/local-read;
        # forward kernels have no UBLKCP -> unchanged -> byte-identical), and the
        # a-operand's def chain actually being the CGA CTA rank (S2UR
        # SR_CgaCtaId — see _prmt_a_is_cluster_rank; a genuine byte-permute
        # that happens to use selector 0x654 falls through to the generic
        # lowering). The rank-strip remap form `PRMT d, RZ, 0x654, addr`
        # (a == RZ) is NOT this pattern (it just clears the rank byte of an
        # existing address) and falls through to the generic byte-permute below.
        if (getattr(self, "_cluster_is_local", False)
                and getattr(self, "_has_ublkcp", False)
                and getattr(sel_op, "is_immediate", False)
                and sel_op.immediate_value == 0x654
                and str(getattr(a_op, "reg", "") or "").upper() not in ("RZ", "URZ")
                and self._prmt_a_is_cluster_rank(Inst, a_op)):
            self._set_val(IRBuilder, IRRegs, dest, self.ir.Constant(i32, 0))
            return

        va = _get_val(a_op, "prmt_a")
        vb = _get_val(b_op, "prmt_b")
        vsel = _get_val(sel_op, "prmt_sel")

        va = self._insert_type_cast_if_needed(IRBuilder, va, i32, "prmt_a")
        vb = self._insert_type_cast_if_needed(IRBuilder, vb, i32, "prmt_b")
        vsel = self._insert_type_cast_if_needed(IRBuilder, vsel, i32, "prmt_sel")

        # Build 8-byte source: {B[3:0], A[3:0]} = (B << 32) | A
        a64 = IRBuilder.zext(va, i64, "prmt_a64")
        b64 = IRBuilder.zext(vb, i64, "prmt_b64")
        b_shifted = IRBuilder.shl(b64, self.ir.Constant(i64, 32), "prmt_bsh")
        src8 = IRBuilder.or_(a64, b_shifted, "prmt_src8")

        # Extract 4 bytes selected by selector nibbles
        result = self.ir.Constant(i32, 0)
        for byte_pos in range(4):
            # Extract selector nibble for this byte
            shift_amt = byte_pos * 4
            nibble = IRBuilder.lshr(vsel, self.ir.Constant(i32, shift_amt), f"prmt_nib{byte_pos}_sh")
            idx = IRBuilder.and_(nibble, self.ir.Constant(i32, 0x7), f"prmt_idx{byte_pos}")
            sign_bit = IRBuilder.and_(nibble, self.ir.Constant(i32, 0x8), f"prmt_sign{byte_pos}")

            # Extract byte from 8-byte source
            idx64 = IRBuilder.zext(idx, i64, f"prmt_idx{byte_pos}_64")
            byte_shift = IRBuilder.shl(idx64, self.ir.Constant(i64, 3), f"prmt_bsh{byte_pos}")
            shifted = IRBuilder.lshr(src8, byte_shift, f"prmt_ext{byte_pos}")
            byte_val = IRBuilder.trunc(shifted, i8, f"prmt_byte{byte_pos}")

            # Sign-extend if bit 3 of selector nibble is set:
            # When bit 3 is set, the output byte is the sign-fill of the
            # selected source byte: 0xFF if bit 7 is set, 0x00 otherwise.
            has_sign = IRBuilder.icmp_unsigned("!=", sign_bit, self.ir.Constant(i32, 0), f"prmt_hassign{byte_pos}")
            zext_val = IRBuilder.zext(byte_val, i32, f"prmt_zext{byte_pos}")
            sign_fill = IRBuilder.ashr(byte_val, self.ir.Constant(i8, 7), f"prmt_sfill{byte_pos}")
            sign_fill32 = IRBuilder.zext(sign_fill, i32, f"prmt_sfill32_{byte_pos}")
            sign_fill_masked = IRBuilder.and_(sign_fill32, self.ir.Constant(i32, 0xFF), f"prmt_smask{byte_pos}")
            selected = IRBuilder.select(has_sign, sign_fill_masked, zext_val, f"prmt_sel{byte_pos}")

            # Place byte into result at correct position
            placed = IRBuilder.shl(selected, self.ir.Constant(i32, byte_pos * 8), f"prmt_place{byte_pos}")
            result = IRBuilder.or_(result, placed, f"prmt_res{byte_pos}")

        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("MATCH")
    def lift_match(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        val = _get_val(src)

        mode = Inst.opcodes[1]
        type_tok = Inst.opcodes[2] if len(Inst.opcodes) > 2 else "U32"

        if mode != "ANY":
            raise UnsupportedInstructionException(f"Unsupported match mode {mode}")

        dtype_map = {"U64": 64, "U32": 32}
        width = dtype_map.get(type_tok)
        if width is None:
            raise UnsupportedInstructionException(f"Unsupported match type {type_tok}")

        mask_const = self.ir.Constant(self.ir.IntType(32), 0xFFFFFFFF)
        intrinsic_name = f"match_any_i{width}"
        match_args = [mask_const, val]
        if self.collective_site_arg:
            match_args.append(self._collective_site(Inst))
        matchVal = self.run_intrinsic(intrinsic_name, IRBuilder, match_args, intrinsic_name)

        # LLVM 14's match.any.sync.i64 declared return type is i64 even
        # though the hardware mask fits in 32 bits; widen or narrow to
        # match the SASS dest register width.
        dest_ty = dest.get_ir_type(self)
        if dest_ty != matchVal.type:
            if dest_ty.width > matchVal.type.width:
                matchVal = IRBuilder.zext(matchVal, dest_ty, "match_any_ext")
            else:
                matchVal = IRBuilder.trunc(matchVal, dest_ty, "match_any_trunc")

        self._set_val(IRBuilder, IRRegs, dest, matchVal)

    @lift_for("MUFU")
    def lift_mufu(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        src = Inst.get_uses()[0]
        func = Inst.opcodes[1] if len(Inst.opcodes) > 1 else None
        v = _get_val(src, "mufu_src")

        if func == "RRO":
            rint_val = self.run_intrinsic("rint_f32", IRBuilder, [v], "rro_rint")
            res = IRBuilder.fsub(v, rint_val, "rro")
            self._set_val(IRBuilder, IRRegs, dest, res)
            return

        # MUFU.TANH: compute tanh(x) = 1 - 2/(exp(2x) + 1)
        # using exp2: exp(2x) = exp2(2x * log2(e))
        if func == "TANH":
            f32 = self.ir.FloatType()
            v_f32 = self._insert_type_cast_if_needed(IRBuilder, v, f32, "tanh_src_f32")
            log2e = self.ir.Constant(f32, 1.4426950408889634)
            two = self.ir.Constant(f32, 2.0)
            one = self.ir.Constant(f32, 1.0)
            # 2x * log2(e)
            two_x = IRBuilder.fmul(two, v_f32, "tanh_2x")
            exp_arg = IRBuilder.fmul(two_x, log2e, "tanh_exp_arg")
            exp_2x = self.run_intrinsic("exp2_f32", IRBuilder, [exp_arg], "tanh_exp2x")
            # exp(2x) + 1
            denom = IRBuilder.fadd(exp_2x, one, "tanh_denom")
            # 2 / (exp(2x) + 1)
            frac = IRBuilder.fdiv(two, denom, "tanh_frac")
            # 1 - 2/(exp(2x) + 1)
            res = IRBuilder.fsub(one, frac, "tanh_result")
            self._set_val(IRBuilder, IRRegs, dest, res)
            return

        mufu_name_map = {
            "RCP": "rcp_f32",
            "RCP64H": "rcp_f64",
            "RSQ": "rsqrt_f32",
            "RSQ64H": "rsqrt_f64",
            "SQRT": "sqrt_f32",
            "SIN": "sin_f32",
            "COS": "cos_f32",
            "EX2": "exp2_f32",
            "LG2": "log2_f32",
        }
        intrinsic_name = mufu_name_map.get(func)
        if intrinsic_name is None:
            raise UnsupportedInstructionException(f"Unknown MUFU function: {func}")
        # SASS MUFU.SIN/COS take the angle in TURNS (period 1.0), so the result
        # is sin(2*pi*x). The llvm.nvvm sin/cos intrinsics take radians (PTXAS
        # re-inserts the 1/(2*pi) reduction), so pre-multiply by 2*pi to cancel
        # it and reproduce MUFU(x) exactly.
        if func in ("SIN", "COS"):
            f32 = self.ir.FloatType()
            v = self._insert_type_cast_if_needed(IRBuilder, v, f32, "mufu_trig_in_f32")
            two_pi = self.ir.Constant(f32, 6.283185307179586)
            v = IRBuilder.fmul(v, two_pi, "mufu_trig_unreduce")

        if func in ("RCP64H", "RSQ64H"):
            tag = func.lower()
            i32 = self.ir.IntType(32)
            i64 = self.ir.IntType(64)
            if src.is_immediate:
                # cuobjdump renders the 32-bit immediate as the f64 VALUE whose
                # upper word it encodes (`MUFU.RCP64H R3, -3` seeds rcp(-3.0);
                # encoding holds 0xC0080000). The lower word is implicit-zero,
                # so the printed value IS the f64 source. Routing it through
                # the integer hi32 reconstruction below would reinterpret -3
                # as 0xFFFFFFFD — NaN space — and poison the Newton-Raphson
                # chain (lci-cuda NaN cluster).
                src_f64 = self.ir.Constant(
                    self.ir.DoubleType(), float(src.immediate_value))
            else:
                # SASS RCP64H/RSQ64H take only the upper 32 bits of an f64;
                # the lower 32 bits are implicit-zero. Construct the full i64
                # by zext-then-shl, then bitcast to double — direct bitcast i32
                # to double is a size mismatch and rejected by llc.
                src_i32 = self._insert_type_cast_if_needed(IRBuilder, v, i32, f"{tag}_src_i32")
                src_i64 = IRBuilder.zext(src_i32, i64, f"{tag}_src_i64")
                src_i64 = IRBuilder.shl(src_i64, self.ir.Constant(i64, 32), f"{tag}_src_shift")
                src_f64 = IRBuilder.bitcast(src_i64, self.ir.DoubleType(), f"{tag}_f64")
            f64_result = self.run_intrinsic(intrinsic_name, IRBuilder, [src_f64], f"{tag}_op")
            as_i64 = IRBuilder.bitcast(f64_result, self.ir.IntType(64), f"{tag}_i64")
            hi32 = IRBuilder.lshr(as_i64, self.ir.Constant(self.ir.IntType(64), 32), f"{tag}_hi_shift")
            hi32_trunc = IRBuilder.trunc(hi32, self.ir.IntType(32), f"{tag}_hi32")
            res = IRBuilder.bitcast(hi32_trunc, self.ir.FloatType(), f"{tag}_result")
        else:
            res = self.run_intrinsic(intrinsic_name, IRBuilder, [v], f"mufu_{func.lower()}")
            # DEBUG (gated): dump the EX2 input/output for the bwd softmax P-recompute
            # to pin why P reads as 1.0. Only when _fa_bwd + env (lift-time gate keeps
            # fwd + non-bwd byte-identical).
            if (func == "EX2" and getattr(self, "_fa_bwd", False)
                    and os.environ.get("SLIFTER_EX2_DUMP")):
                f32 = self.ir.FloatType()
                vin = self._insert_type_cast_if_needed(IRBuilder, v, f32, "ex2dbg_in")
                self.run_intrinsic("slifter_ex2_dump", IRBuilder, [vin, res], "ex2dbg")
        self._set_val(IRBuilder, IRRegs, dest, res)

    @lift_for("QSPC")
    def lift_qspc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        addr_op = Inst.get_uses()[-1]
        
        addr_val = _get_val(addr_op, "qspc_addr")

        space_map = {
            "S": "isspacep_shared",
            "G": "isspacep_global",
            "L": "isspacep_local",
            "C": "isspacep_const",
        }

        space = Inst.opcodes[2]
        intrinsic_name = space_map[space]

        # The intrinsic takes i8* in LLVM 14 (see lifter_nvvm.run_intrinsic);
        # the SASS operand is an i64 register value, so widen-and-cast before
        # the call.
        i64 = self.ir.IntType(64)
        i8ptr = self.ir.PointerType(self.ir.IntType(8))
        addr_val = self._insert_type_cast_if_needed(IRBuilder, addr_val, i64, "qspc_addr_i64")
        addr_ptr = IRBuilder.inttoptr(addr_val, i8ptr, "qspc_addr_ptr")

        result = self.run_intrinsic(intrinsic_name, IRBuilder, [addr_ptr], f"qspc_{space.lower()}")
        
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("LEPC")
    def lift_lepc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        dest = Inst.get_defs()[0]
        result = self.ir.Constant(self.ir.IntType(64), 0)
        self._set_val(IRBuilder, IRRegs, dest, result)

    @lift_for("CALL")
    def lift_call(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        # SASS CALL.REL.NOINC <addr> in cuBLAS / CUTLASS commonly encodes a
        # forward branch to inline epilogue code that itself terminates the
        # kernel (BUG-091 pattern B: target ends with EXIT, never RET).
        # Treat CALL with a literal target as a plain branch to the target
        # BB. The downstream EXIT inside the body is already lifted to
        # ret_void; if the target uses RET (real subroutine, pattern A),
        # we'd need to inline -- not yet handled.
        target_op = Inst.get_uses()[0]
        if target_op.is_reg or target_op.is_reg_group:
            if Inst.parent and Inst.parent.succs:
                IRBuilder.branch(block_map[Inst.parent.succs[0]])
            return
        # Literal address target: look up the BB at that address.
        target_addr = None
        if target_op.is_immediate:
            target_addr = target_op.immediate_value
        elif target_op.is_mem_addr:
            target_addr = target_op.offset_value
        # A literal target of 0x0 is an UNRESOLVED relocation placeholder, not a
        # real branch destination: CALL.ABS.NOINC encodes its absolute target in
        # a MOV reloc pair the linker patches, leaving 0x0 in the CALL itself.
        # The BB at addr 0 is the function ENTRY; branching to it makes the entry
        # acquire a predecessor -> llc "Entry block must not have predecessors!"
        # (fresnel/merkle). Treat as an opaque no-op device call and fall through
        # to the successor, preserving control flow.
        if target_addr == 0:
            if Inst.parent and Inst.parent.succs:
                IRBuilder.branch(block_map[Inst.parent.succs[0]])
            else:
                IRBuilder.ret_void()
            return
        if target_addr is not None and block_map is not None:
            target_bb = None
            for bb in block_map.keys():
                if getattr(bb, "addr", None) == target_addr:
                    target_bb = bb
                    break
            if target_bb is not None:
                IRBuilder.branch(block_map[target_bb])
                return
        raise UnsupportedInstructionException(f"Direct CALL not supported: {Inst}")

    @lift_for("CALLFUNC")
    def lift_callfunc(self, Inst, IRBuilder, IRRegs, block_map=None, _get_val=None, **kwargs):
        target_op = Inst.get_uses()[0] if Inst.get_uses() else None
        if target_op is None:
            raise UnsupportedInstructionException(f"CALLFUNC missing target: {Inst}")

        callee_name = target_op.name or str(target_op)
        callee = self.ir_functions.get(callee_name)
        if callee is None:
            fn_ty = self.ir.FunctionType(self.ir.VoidType(), [], False)
            callee = self.ir.Function(self.llvm_module, fn_ty, callee_name)
            self.ir_functions[callee_name] = callee

        IRBuilder.call(callee, [])

    def _set_val(self, IRBuilder, IRRegs, operand, val):
        if operand.is_reg_group:
            reg_count = len(operand.reg_group)
            total_bits = reg_count * 32
            name = operand.name or "reg_group"
            val = self._insert_type_cast_if_needed(IRBuilder, val, self.ir.IntType(total_bits), name)
            for i, sub_op in enumerate(operand.reg_group):
                if i == 0:
                    sub_val = IRBuilder.trunc(val, self.ir.IntType(32), f"{name}_r{i}")
                else:
                    shifted = IRBuilder.lshr(
                        val,
                        self.ir.Constant(self.ir.IntType(total_bits), i * 32),
                        f"{name}_r{i}_shift",
                    )
                    sub_val = IRBuilder.trunc(shifted, self.ir.IntType(32), f"{name}_r{i}")
                IRRegs[sub_op.get_ir_name(self)] = sub_val
            return

        target_ty = operand.get_ir_type(self)
        # BUG-082: predicate registers (i1) stored directly into IRRegs leak
        # cross-BB i1 references that violate SSA dominance when the SSA pass
        # fails to insert a PHI at the consumer's block (an underlying issue
        # in the SSA / RegRemap pipeline that is too invasive to fix in this
        # bug). Coerce predicate stores through an i32 carrier in the current
        # block: the i32 flows through the SSA PHI machinery (predicate-typed
        # PHIs are equally susceptible, but TypeAnalysis often leaves the
        # destination as NOTYPE → i32 already, so we just normalize).
        # Consumers (lift_isetp combiner, lift_pbra, lift_sel, etc.) use
        # `_normalize_i1_in_curr_bb` to convert back to i1 in their own BB.
        i1 = self.ir.IntType(1)
        if target_ty == i1 and operand.is_predicate_reg:
            i32 = self.ir.IntType(32)
            if val.type == i1:
                val = IRBuilder.zext(val, i32, f"{operand.name or 'pred'}_to_i32")
            else:
                val = self._insert_type_cast_if_needed(IRBuilder, val, i32, operand.name)
        else:
            val = self._insert_type_cast_if_needed(IRBuilder, val, target_ty, operand.name)
        IRRegs[operand.get_ir_name(self)] = val

    def get_ir_type(self, type_desc):
        # Scalar integer / boolean types — signed and unsigned share the same
        # LLVM IR representation (IntType(N)); signedness is an op-level
        # property, not a type-level one.
        if type_desc in INT8 or type_desc in UINT8:
            return self.ir.IntType(8)
        if type_desc in INT16 or type_desc in UINT16:
            return self.ir.IntType(16)
        if type_desc in INT32 or type_desc in UINT32:
            return self.ir.IntType(32)
        if type_desc in INT64 or type_desc in UINT64:
            return self.ir.IntType(64)
        if type_desc in INT128:
            return self.ir.IntType(128)
        if type_desc in BOOL:
            return self.ir.IntType(1)
        # Floating-point types
        if type_desc in FLOAT16:
            return self.ir.HalfType()
        if type_desc in BF16:
            # SASS general registers are 32-bit; BF16 values are stored as
            # a single bf16 in the low 16 bits or as packed bf16x2. Use i32
            # storage so packed atomics (ATOMG.CAS for f16x2/bf16x2 atomic
            # add) see the full 32-bit value and the BF16_V2 lift paths
            # (HFMA2/HADD2/HMUL2) can bitcast to `<2 x bfloat>` for native
            # LLVM bf16 arithmetic. Tensor-core HMMA.BF16 paths likewise
            # construct `<2 x bfloat>` via _half2_elem_ty for intrinsic args.
            return self.ir.IntType(32)
        if type_desc in FLOAT32 or type_desc in TF32:
            # TF32 is a 19-bit-in-32-bit tensor-float precision; LLVM has
            # no native tf32 IR type so it lowers to f32 (upper 19 bits
            # carry the value, lower 13 are zeroed by the tensor core).
            return self.ir.FloatType()
        if type_desc in FLOAT64:
            return self.ir.DoubleType()
        # Pointer-qualified tags — still raw strings; no lattice constant
        # because they're not propagated by TypeAnalysis, only emitted at
        # addr-of sites by the lifter.
        if type_desc == "Int32_PTR":
            return self.ir.PointerType(self.ir.IntType(32), 1)
        if type_desc == "Float32_PTR":
            return self.ir.PointerType(self.ir.FloatType(), 1)
        if type_desc == "Int64_PTR":
            return self.ir.PointerType(self.ir.IntType(64), 1)
        if type_desc == "Int128_PTR":
            return self.ir.PointerType(self.ir.IntType(128), 1)
        if type_desc == "Float64_PTR":
            return self.ir.PointerType(self.ir.DoubleType(), 1)
        if type_desc == "PTR":
            return self.ir.PointerType(self.ir.IntType(8), 1)
        if type_desc == "NOTYPE":
            return self.ir.IntType(32)

        raise ValueError(f"Unknown type: {type_desc}")

    def shutdown(self):
        binding.shutdown()
