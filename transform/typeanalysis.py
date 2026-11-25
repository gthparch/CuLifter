from transform.transform import SaSSTransform
from sir.operand import Operand
from collections import deque
from sir.instruction import Instruction
import os

# Type-tag convention: each UPPERCASE Python name is a frozenset holding
# the MixedCase string tag(s) for that lattice element. Use `tag in NAME`
# for membership checks, never raw string equality — that's the whole
# point of having a constant. Unsigned int tags ("UInt32", "UInt64") are
# *not* members of the signed INT* sets because downstream IR typing may
# want to distinguish them; UINT* below are provided so the string tags
# have single sources of truth.
BOOL = frozenset({"Bool"})
FLOAT16 = frozenset({"Float16"})
BF16 = frozenset({"BF16"})
# TF32 — 19-bit tensor-float (1 sign + 8 exp + 10 mantissa), used as the
# A/B input precision on HMMA.1688.F32.TF32. Storage is 32-bit (upper 19
# bits carry the value; lower 13 bits are effectively zeroed by the
# hardware). Treated as FLOAT32 at the LLVM IR level (no native tf32 in
# llvmlite) but kept as a distinct lattice element so the type-flow
# analysis can distinguish tensor-core precision from plain Float32.
TF32 = frozenset({"TF32"})
FLOAT32 = frozenset({"Float32"})
FLOAT64 = frozenset({"Float64"})
INT8 = frozenset({"Int8"})
INT16 = frozenset({"Int16"})
INT32 = frozenset({"Int32"})
INT64 = frozenset({"Int64"})
INT128 = frozenset({"Int128"})
UINT8 = frozenset({"UInt8"})
UINT16 = frozenset({"UInt16"})
UINT32 = frozenset({"UInt32"})
UINT64 = frozenset({"UInt64"})
INT = INT32 | INT64 | INT128
NUM1 = BOOL
# 16-bit bag: F16 and BF16 are both 16 bits wide but with different
# exponent/mantissa splits. For register-level width matching (PACK64,
# MOV, LDG.16 propagation) they are interchangeable; the instruction's
# opcode (e.g. HFMA2 vs HFMA2.BF16_V2) disambiguates at seeding time.
NUM16 = FLOAT16 | BF16
# TF32 sits in the 32-bit bag: register storage is 32-bit and same-width
# ops (MOV, PACK64 halves, LDG.32) freely interchange with Int32/Float32.
# Tensor-core ops re-narrow it via the HMMA seed table.
NUM32 = INT32 | FLOAT32 | TF32
NUM64 = INT64 | FLOAT64
NUM128 = INT128
TOP = NUM32 | NUM1 | NUM64 | NUM128 | NUM16
BOTTOM = frozenset()

# Sentinel for _store_op_type (None/empty-set are valid stored values).
_MISSING = object()

# Default type tag used when TypeAnalysis cannot resolve an operand.
# Exposed as a named constant so the "what do we fall back to?" choice
# lives in one place rather than being a magic string at every site.
DEFAULT_UNTYPED = "UInt32"

class TypeAnalysis(SaSSTransform):

    # Cost-evaluation instrumentation (default off — zero production impact).
    # When RECORD_ITERS is True, every process_func appends (func_name,
    # n_fixpoint_iterations) to ITERATION_LOG for profiling callers.
    RECORD_ITERS = False
    ITERATION_LOG = []

    def __init__(self, max_iterations=None):
        super().__init__()
        # max_iterations: None = normal (up to 5), 0 = seed-only (no propagation)
        self.max_iterations = max_iterations

        # Read once: per-instruction os.environ.get cost ~15s on 640k-inst
        # modules (cuDNN Ops so.75 profile).
        _bl = os.environ.get("SLIFTER_SEED_BLACKLIST")
        self._seed_blacklist = frozenset(_bl.split(",")) if _bl else frozenset()

        self.instruction_type_table = {
            "S2R": [[INT], [INT, INT]],
            "IMAD": [[INT32], [INT32, INT32, INT32]],
            "SDIV": [[INT32], [INT32, INT32]],
            # UREM64/UREM64HI: IntDivRecognize's u64 modulo pseudo-ops —
            # low/high word of (Dhi:Dlo % dhi:dlo); uses are the four words.
            "UREM64": [[INT32], [INT32, INT32, INT32, INT32]],
            "UREM64HI": [[INT32], [INT32, INT32, INT32, INT32]],
            # UDIV64/UDIV64HI: the u64 *quotient* siblings (45_dual_gemm's
            # coord=index/N divide) — low/high word of (Dhi:Dlo / dhi:dlo);
            # same four-word operand layout.
            "UDIV64": [[INT32], [INT32, INT32, INT32, INT32]],
            "UDIV64HI": [[INT32], [INT32, INT32, INT32, INT32]],
            "IMAD.WIDE": [[INT64], [INT32, INT32, INT64]],
            # sm70+ IMAD.HI: addend register names a 64-bit pair {Rc+1:Rc}
            # added before hi-extraction (paired in sir/instruction.py).
            "IMAD.HI": [[INT32], [INT32, INT32, INT64]],
            "IMAD.MOV": [[NUM32], [INT32, INT32, INT32]],
            "IADD3": [[INT32, BOOL], [INT32, INT32, INT32]],
            "XMAD": [[INT32], [INT32, INT32, INT32]],
            "IADD32I": [[INT32], [INT32, INT32]],
            "IADD": [[INT32], [INT32, INT32]],
            "ISETP": [[BOOL, BOOL], [INT32, INT32, BOOL, BOOL]],
            "LEA": [[INT32, BOOL], [INT32, INT32, INT32]],
            "LEA.HI.X": [[INT32], [INT32, INT32, INT32, INT32, BOOL]],
            "LEA.X": [[INT32], [INT32, INT32, INT32, BOOL]],
            "LEA.HI": [[INT32], [INT32, INT32, INT32, INT32]],
            "LEA.HI.SX32": [[INT32], [INT32, INT32, INT32]],
            "LEA.HI.X.SX32": [[INT32], [INT32, INT32, INT32, BOOL]],
            "F2I": [[INT32], [FLOAT32]],
            "UF2I": [[INT32], [FLOAT32]],
            "F2I.U64": [[INT64], [FLOAT32]],
            "F2I.S64": [[INT64], [FLOAT32]],
            "F2I.F64": [[INT32], [FLOAT64]],
            "F2I.U64.F64": [[INT64], [FLOAT64]],
            "F2I.S64.F64": [[INT64], [FLOAT64]],
            "I2F": [[FLOAT32], [INT32]],
            "I2F.F64": [[FLOAT64], [INT]],
            "I2F.S64": [[FLOAT32], [INT64]],
            "I2F.U64": [[FLOAT32], [INT64]],
            "I2F.F64.S64": [[FLOAT64], [INT64]],
            "I2F.F64.U64": [[FLOAT64], [INT64]],
            "F2F": [[TOP], [TOP]],
            "F2F.F64": [[TOP], [TOP]],
            "F2FP": [[INT32], [FLOAT32, FLOAT32]],
            "F2FP.PACK_AB_MERGE_C": [[INT32], [FLOAT32, FLOAT32, INT32]],
            # UNPACK_B: fp8x2 (packed bytes, integer) -> f16x2 (packed i32)
            "F2FP.UNPACK_B": [[INT32], [INT32]],
            "F2FP.UNPACK_A": [[INT32], [INT32]],
            # F2IP.S8.F32: float->int8 pack-pair (cvt.pack.sat.s8.s32.b32 fused with
            # the float convert). Rd = packed int8x... in an i32; Ra,Rb are the two
            # F32 sources, Rc the i32 accumulator (RZ on the first call).
            "F2IP": [[INT32], [FLOAT32, FLOAT32, INT32]],
            "SEL": [[INT32], [INT32, INT32, BOOL]],
            "NOP": [[], []],
            "BRA": [[], [TOP, INT32]],
            "CALL": [[], [TOP]],
            "CALLFUNC": [[], [TOP]],
            "EXIT": [[], []],
            "RET": [[], []],
            "SYNC": [[], []],
            "WARPSYNC": [[], [INT32]],
            "BPT": [[], [INT32]],
            "BAR": [[], [INT32]],
            "BAR.RED.AND": [[], [INT32, BOOL]],
            "BAR.RED.OR": [[], [INT32, BOOL]],
            "BAR.RED.POPC": [[], [INT32, INT32]],
            "B2R": [[INT32, BOOL], []],
            # P2R packs P0-P6 into an integer register; R2P unpacks integer bits into P0-P6
            # P2R Rd, PR, Rs, mask → def: Rd (i32), uses: PR (pseudo), Rs (i32), mask (imm)
            "P2R": [[INT32], [INT32, INT32, INT32]],
            # R2P PR, Rs, mask → def: PR (pseudo), uses: Rs (i32), mask (imm)
            "R2P": [[], [INT32, INT32]],
            "SSY": [[], []],
            "DEPBAR": [[], []],
            # SM90 NOP opcodes (handled by x86 lifter as no-ops)
            # uses[0] is the shared-memory byte address of the mbarrier — pin it to
            # Int32 like every real memory op (LDS/STS/ATOMS/LDGSTS) so a PHI merging
            # the predicated-arrive address doesn't unify with a Float32 wgmma
            # accumulator that ptxas parked in the same physical reg (the split-kernel
            # nvvm lift crash: `phi float` -> _shared_gep GEP-index TypeError).
            "SYNCS": [[], [INT32]],
            "SYNCS.EXCH.64": [[], [INT32, INT64]],
            "FENCE": [[], []],
            "ELECT": [[BOOL, INT32], [BOOL]],
            # IDP.4A.S8.S8 Rd, Ra, Rb, Rc — integer dot product
            "IDP": [[INT32], [INT32, INT32, INT32]],
            "NANOSLEEP": [[], []],
            "STAS": [[], [INT32, INT32, INT32]],
            "REDAS": [[], []],
            "WARPGROUP": [[], []],
            "LOP32I": [[INT32], [INT32, INT32]],
            "ISCADD": [[INT32], [INT32, INT32]],
            "MOV32I": [[INT32], [INT32]],
            "IABS": [[INT32], [INT32]],
            "MATCH": [[INT32], [INT32]],
            "MATCH.U32": [[TOP], [INT32]],
            "MATCH.U64": [[TOP], [INT64]],
            "RED": [[], [INT64, INT32]],
            "IMNMX": [[INT32], [INT32, INT32, BOOL]],
            "IMNMX.U32": [[INT32], [INT32, INT32]],
            "IMNMX.U64": [[INT64], [INT64, INT64]],
            "MOV": [[NUM32], [NUM32]],

            "ATOMS": [[INT32], [INT32, NUM32, NUM32]],
            "ATOMS.64": [[INT32], [INT32, NUM64, NUM64]],
            "ATOM": [[INT32], [INT32, NUM32, NUM32]],
            
            # Bit manipulation (mostly type agnostic)
            # "SHF": [[NUM32], [NUM32, NUM32, NUM32]],
            # "LOP3": [[NUM32 | NUM1, NUM32], [NUM32, NUM32, NUM32, INT32, BOOL]],
            # "PRMT": [[NUM32], [NUM32, NUM32, INT32]],
            # "POPC": [[INT32], [NUM32]],
            # "SHL": [[NUM32], [NUM32, NUM32, NUM32]],
            # "BREV": [[NUM32], [NUM32]],
            # "FLO": [[INT32], [NUM32]],
            
            "SHF": [[INT32], [INT32, INT32, INT32]],
            "PRMT": [[INT32], [INT32, INT32, INT32]],
            "LOP3": [[INT32, INT32], [INT32, INT32, INT32, INT32, BOOL]],
            "PRMT": [[INT32], [INT32, INT32, INT32]],
            "POPC": [[INT32], [INT32]],
            "SHL": [[INT32], [INT32, INT32, INT32]],
            "BREV": [[INT32], [INT32]],
            "FLO": [[INT32], [INT32]],
            "SGXT": [[INT32], [INT32, INT32]],
            
            # Predicate instructions
            "PLOP3": [[BOOL, BOOL], [BOOL, BOOL, BOOL, INT32, BOOL]],
                        
            # Memory
            "QSPC": [[BOOL], [INT64, INT64]],
            "LDS": [[NUM32], [INT32]],
            "LDS.64": [[NUM64], [INT32]],
            "LDS.128": [[NUM128], [INT32]],
            "STS": [[], [INT32, NUM32]],
            "STS.64": [[], [INT32, NUM64]],
            "STS.128": [[], [INT32, NUM128]],
            "LDG": [[NUM32], [INT64]],
            "LDG.64": [[NUM64], [INT64]],
            "LDG.128": [[NUM128], [INT64]],
            "STG": [[], [INT64, NUM32]],
            "STG.64": [[], [INT64, NUM64]],
            "STG.128": [[], [INT64, NUM128]],
            "LD": [[NUM32], [INT64]],
            "LD.64": [[NUM64], [INT64]],
            "LD.128": [[NUM128], [INT64]],
            "ST": [[], [INT64, NUM32]],
            "ST.64": [[], [INT64, NUM64]],
            "ST.128": [[], [INT64, NUM128]],
            "LDL": [[NUM32], [INT32]],
            "LDL.64": [[NUM64], [INT32]],
            "LDL.128": [[NUM128], [INT32]],
            "STL": [[], [INT32, NUM32]],
            "STL.64": [[], [INT32, NUM64]],
            "STL.128": [[], [INT32, NUM128]],
            "LDC": [[NUM32], [INT64]],
            "LDC.64": [[NUM64], [INT64]],

            # SM90 async copy: LDGSTS.E [shared_off], desc[UR][global_addr.64]
            # No defs (classified as store). uses[0]=shared offset (INT32), uses[1]=global addr (INT64)
            "LDGSTS": [[], [INT32, INT64]],

            # SM90 shared memory matrix tile load/store (LDSM/STSM)
            # LDSM: 1 def (loaded data), 1 use (shared addr INT32)
            "LDSM": [[NUM32], [INT32]],
            # STSM: no defs, uses[0]=shared addr (INT32), uses[1]=data
            "STSM": [[], [INT32, NUM32]],

            # SM90 global atomic with descriptor: ATOMG.E.op desc[UR][R.64], Rsrc
            # 2 defs (predicate + old value), uses: addr(INT64) + src(NUM32) [+ CAS operand]
            "ATOMG": [[BOOL, NUM32], [INT64, NUM32, NUM32]],
            # SM90 global reduction with descriptor: REDG.E.op desc[UR][R.64], Rsrc
            # No defs, uses: addr(INT64) + src(NUM32)
            "REDG": [[], [INT64, NUM32]],

            # Half operations
            # For now we treat half2 as int32
            # they are casted at lift stage
            "HMMA": [[FLOAT32] * 4, [FLOAT32] * 7],
            # Synthetic op produced by WmmaPeephole. Operand counts and
            # element types are variant-dependent (shape × accum); the
            # entry below covers the M16N16K16 f16→f32 case (16 f32 D
            # regs, 4 A v2half + 4 B v2half + 8 C f32). New variants get
            # their own entries by full opcode key like the HMMA family.
            "WMMA_MMA.M16N16K16": [[FLOAT32] * 16, [FLOAT16] * 8 + [FLOAT32] * 8],
            "HMMA.F32": [[FLOAT32] * 4, [FLOAT32] * 7],
            "HMMA.F16": [[FLOAT16] * 4, [FLOAT16] * 8],
            # HMMA.16816.F32: 4 defs, 4 A regs + 2 B regs + 4 C regs = 10 uses
            "HMMA.16816.F32": [[FLOAT32] * 4, [FLOAT32] * 10],
            # HMMA.16816.F16: 2 defs, 4 A regs + 2 B regs + 2 C regs = 8 uses.
            # Every register is a PACKED v2f16 pair (32 bits). Typing them
            # FLOAT16 made _get_val trunc the i32 reg to 16 bits before the
            # shim's v2f16 cast zero-padded it back — the hi half of every
            # fragment was DISCARDED (hgemm: odd output columns identically
            # zero + ~2x undersum from the missing k-products). FLOAT32 is
            # width-true, mirroring the working F32-accum row above; the
            # shim bitcasts 32-bit values to v2f16 losslessly.
            "HMMA.16816.F16": [[FLOAT32] * 2, [FLOAT32] * 8],
            # HMMA.16816.F32.BF16: F32 accumulator, BF16 inputs. Shape is
            # identical to the F32 variant (4 defs / 10 uses) but the first
            # 6 uses (A:4 + B:2) are packed 2x bfloat16 instead of 2x half.
            # The 4 accumulator uses and 4 defs stay F32.
            "HMMA.16816.F32.BF16": [[FLOAT32] * 4, [BF16] * 6 + [FLOAT32] * 4],
            # HMMA.1688.F16: m16n8k8 F16 accumulator. After implicit-reg
            # expansion (sir/instruction.py): D=2 v2f16, A=2 v2f16, B=1 v2f16,
            # C=2 v2f16 → 2 defs / 5 uses. Every register is a PACKED v2f16
            # pair (32 bits) — FLOAT32 is width-true, same trap as the
            # HMMA.16816.F16 row above (FLOAT16 typing trunc'd each reg to
            # its lo half: odd output columns identically zero + undersum
            # from the missing odd-k products; hmma1688f16 fixture 126/128).
            "HMMA.1688.F16": [[FLOAT32] * 2, [FLOAT32] * 5],
            # HMMA.1688.F32.BF16: m16n8k8 with F32 accumulator and BF16 inputs.
            # D=4 f32, A=2 v2bfloat, B=1 v2bfloat, C=4 f32 → 4 defs / 7 uses.
            "HMMA.1688.F32.BF16": [[FLOAT32] * 4, [BF16] * 3 + [FLOAT32] * 4],
            # HMMA.1688.F32.TF32: m16n8k8 with F32 accumulator and TF32 inputs.
            # Each input register holds ONE tf32 value (full 32-bit reg, low
            # 13 mantissa bits ignored by the tensor core), so unlike the
            # packed-16-bit F16/BF16 forms A needs 4 regs and B needs 2 (cute
            # MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>): 4 defs / 10 uses.
            # Typing A/B as TF32 distinguishes them from plain Float32 in the
            # lattice; LLVM IR emits f32 either way.
            "HMMA.1688.F32.TF32": [[FLOAT32] * 4, [TF32] * 6 + [FLOAT32] * 4],
            # Sparse TF32 also carries a full B fragment and one metadata register.
            "HMMA.SP.16816.F32.TF32": [[FLOAT32] * 4, [TF32] * 8 + [FLOAT32] * 4 + [INT32]],
            # Each A or B register is two packed half-precision values. E is a raw metadata word.
            "HMMA.SP.16832.F32": [[FLOAT32] * 4, [INT32] * 8 + [FLOAT32] * 4 + [INT32]],
            "HMMA.SP.16832.F32.BF16": [[FLOAT32] * 4, [INT32] * 8 + [FLOAT32] * 4 + [INT32]],
            # HGMMA/WGMMA: warpgroup MMA (SM90)
            # Note: the dynamic handler in resolve_types() intercepts these
            # opcodes and sizes types from actual operand count. The static
            # entries below are fallback documentation for reference only.
            # Register form: D=N/2 defs(F32), uses = A(4)+B(N/4)+C(N/2)
            "HGMMA.64x8x16.F32": [[FLOAT32] * 4, [FLOAT32] * 10],
            "HGMMA.64x16x16.F32": [[FLOAT32] * 8, [FLOAT32] * 16],
            "HGMMA.64x32x16.F32": [[FLOAT32] * 16, [FLOAT32] * 28],
            "HGMMA.64x64x16.F32": [[FLOAT32] * 32, [FLOAT32] * 52],
            "HGMMA.64x128x16.F32": [[FLOAT32] * 64, [FLOAT32] * 100],
            "HGMMA.64x256x16.F32": [[FLOAT32] * 128, [FLOAT32] * 196],
            # F16 accumulation: D=N/4 defs
            "HGMMA.64x8x16.F16": [[FLOAT16] * 2, [FLOAT16] * 8],
            "HGMMA.64x16x16.F16": [[FLOAT16] * 4, [FLOAT16] * 12],
            "HGMMA.64x32x16.F16": [[FLOAT16] * 8, [FLOAT16] * 20],
            "HGMMA.64x64x16.F16": [[FLOAT16] * 16, [FLOAT16] * 36],
            "HGMMA.64x128x16.F16": [[FLOAT16] * 32, [FLOAT16] * 68],
            "HGMMA.64x256x16.F16": [[FLOAT16] * 64, [FLOAT16] * 132],
            # Generic HGMMA.F32 fallback for descriptor forms
            "HGMMA.F32": [[FLOAT32] * 4, [FLOAT32] * 10],
            # Generic HGMMA.F16 fallback
            "HGMMA.F16": [[FLOAT16] * 2, [FLOAT16] * 8],
            # WGMMA aliases (same as HGMMA)
            "WGMMA.F32": [[FLOAT32] * 4, [FLOAT32] * 10],
            "WGMMA.F16": [[FLOAT16] * 2, [FLOAT16] * 8],
            "HSETP2": [[BOOL, BOOL], [INT32, INT32, BOOL]],
            "HFMA2": [[INT32], [INT32, INT32, INT32]],
            # BF16_V2: register is 32-bit (two packed BF16) but the semantic
            # type of both defs and uses is BF16. Distinguishing from INT32
            # lets paper lattice / type-composition stats count these as
            # floating-point values, not integer bag-of-bits.
            "HFMA2.BF16_V2": [[BF16], [BF16, BF16, BF16]],
            
            # Warp-level primitives
            "SHFL": [[BOOL, NUM32], [NUM32, INT32, INT32]],
            "VOTE": [[INT32], [BOOL, BOOL]],
            "VOTEU": [[INT32], [BOOL, BOOL]],
            
            "MOV64": [[NUM64], [NUM64]],
            
            # Float instruction types
            "FADD": [[FLOAT32], [FLOAT32, FLOAT32]],
            "UFADD": [[FLOAT32], [FLOAT32, FLOAT32]],
            "FFMA": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32]],
            "UFFMA": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32]],
            "FMUL": [[FLOAT32], [FLOAT32, FLOAT32]],
            "UFMUL": [[FLOAT32], [FLOAT32, FLOAT32]],
            "FSETP": [[BOOL, BOOL], [FLOAT32, FLOAT32, BOOL]],
            "UFSETP": [[BOOL, BOOL], [FLOAT32, FLOAT32, BOOL]],
            "FSEL": [[FLOAT32], [FLOAT32, FLOAT32, BOOL]],
            "UFSEL": [[FLOAT32], [FLOAT32, FLOAT32, BOOL]],
            "FSET": [[FLOAT32], [FLOAT32, FLOAT32, BOOL]],
            "MUFU": [[FLOAT32], [FLOAT32]],
            "MUFU.RCP64H": [[FLOAT32], [FLOAT64]],
            "FCHK": [[BOOL], [FLOAT32, FLOAT32]],
            "FMNMX": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32, BOOL]],
            "FRND": [[FLOAT32], [FLOAT32]],
            "UFRND": [[FLOAT32], [FLOAT32]],
            "FRND.F64": [[FLOAT64], [FLOAT64]],
            
            # Double instruction types
            "DADD": [[FLOAT64], [FLOAT64, FLOAT64]],
            "DMUL": [[FLOAT64], [FLOAT64, FLOAT64]],
            "DFMA": [[FLOAT64], [FLOAT64, FLOAT64, FLOAT64]],
            "DSETP": [[BOOL, BOOL], [FLOAT64, FLOAT64, BOOL]],
            
            # Half instruction types
            "HADD2": [[INT32], [INT32, INT32]],
            "HADD2.BF16_V2": [[BF16], [BF16, BF16]],

            # Uniform variants
            "S2UR": [[INT32], [INT32, INT32]],
            "ULDC": [[NUM32], [INT32]],
            "ULDC.64": [[NUM64], [INT32]],
            "LDCU": [[NUM32], [INT32]],
            "LDCU.64": [[NUM64], [INT32]],
            "FMNMX3": [[FLOAT32], [FLOAT32, FLOAT32, FLOAT32, BOOL]],
            "USHF": [[INT32], [INT32, INT32, INT32]],
            "ULEA": [[INT32], [INT32, INT32, INT32, INT32]],
            "ULOP3": [[INT32, INT32], [INT32, INT32, INT32, INT32, BOOL]],
            "UPLOP3": [[BOOL, BOOL], [BOOL, BOOL, BOOL, INT32, BOOL]],
            "UIADD3": [[INT32], [INT32, INT32, INT32]],
            "UIMAD": [[INT32], [INT32, INT32, INT32]],
            "UIMAD.WIDE": [[INT64], [INT32, INT32, INT64]],
            "UIMAD.HI": [[INT32], [INT32, INT32, INT64]],
            "UMOV": [[INT32], [INT32]],
            "R2UR": [[INT32], [INT32]],
            "R2UR.BROADCAST": [[INT32], [INT32, BOOL, INT32]],
            "UISETP": [[BOOL, BOOL], [INT32, INT32, BOOL]],
            "USEL": [[INT32], [INT32, INT32, BOOL]],
            
            "BRA": [[], [INT32]],

            # Dummy instruction types
            "PACK64": [[NUM64], [NUM32, NUM32]],
            "UNPACK64": [[NUM32, NUM32], [NUM64]],
            "CAST64": [[NUM64], [NUM32]],
            "IADD64": [[INT64], [INT64, INT64]],
            "IMAD64": [[INT64], [INT32, INT32, INT64]],
            "SHL64": [[INT64], [INT64, INT64, INT64]],
            "IADD32I64": [[INT64], [INT64, INT32]],
            "BITCAST": [[TOP], [TOP]],
            "PBRA": [[], [BOOL, INT32, INT32]],
            "LEA64": [[INT64], [INT64, INT64, INT64]],
            "SETZERO": [[TOP], []],
            "ULDC64": [[NUM64], [INT32]],
            "LDCU64": [[NUM64], [INT32]],
            "LDG64": [[NUM64], [INT64]],
            "SHR64": [[INT64], [INT64, INT64]],
            "ISETP64":  [[BOOL, BOOL], [INT64, INT64, BOOL]],
            "IADD364": [[INT64], [INT64, INT64, INT64]],

            # ── Missing integer ALU ──────────────────────────────────
            "ISUB": [[INT32], [INT32, INT32]],
            "BMSK": [[INT32], [INT32, INT32]],
            "VIADD": [[INT32], [INT32, INT32]],
            "VIMNMX": [[INT32], [INT32, INT32, BOOL]],
            "UVIMNMX": [[INT32], [INT32, INT32, BOOL]],
            "VIMNMX3": [[INT32], [INT32, INT32, INT32]],
            "VIADDMNMX": [[INT32], [INT32, INT32, INT32, BOOL]],
            "VABSDIFF4": [[INT32], [INT32, INT32, INT32]],

            # ── Missing FP16 packed ──────────────────────────────────
            "HMUL2": [[INT32], [INT32, INT32]],
            "HMUL2.BF16_V2": [[BF16], [BF16, BF16]],
            "HMNMX2": [[INT32], [INT32, INT32, BOOL]],
            "HSET2": [[INT32], [INT32, INT32, BOOL, BOOL]],

            # ── Missing conversion ───────────────────────────────────
            # I2FP: integer to float packed (same handler as I2F)
            "I2FP": [[FLOAT32], [INT32]],
            "UI2F": [[FLOAT32], [INT32]],
            "UI2FP": [[FLOAT32], [INT32]],
            "I2FP.F32.S32": [[FLOAT32], [INT32]],
            "I2FP.F32.U32": [[FLOAT32], [INT32]],
            "I2FP.F32.S16": [[FLOAT32], [INT32]],
            "I2FP.F32.U16": [[FLOAT32], [INT32]],
            "I2FP.F32.S8": [[FLOAT32], [INT32]],
            "I2FP.F32.U8": [[FLOAT32], [INT32]],
            # I2IP: integer insert-pack
            "I2IP": [[INT32], [INT32, INT32]],

            # ── Missing tensor core ──────────────────────────────────
            # IMMA: integer matrix multiply-accumulate
            # Shapes: 8816, 8832, 16816, 16832; types: S8/U8/S4/U4
            # Result is struct of 8 x i32; dynamic operand count
            "IMMA": [[INT32] * 8, [INT32] * 8],
            # IMMA.SP.16864.S8.S8[.SAT] — sparse 2:4 integer MMA on m16n8k64.
            # After implicit-reg expansion: 4 defs (D), 4 A + 4 B + 4 C
            # register uses, plus a metadata register and a selector
            # immediate (not register-typed). All integer-typed.
            "IMMA.SP.16864.S8.S8": [[INT32] * 4, [INT32] * 13],
            "IMMA.SP.16864.S8.S8.SAT": [[INT32] * 4, [INT32] * 13],
            # DMMA — F64 tensor-core MMA. After reg-pair expansion all
            # operands are F64 pairs (each a 2×r32 pair of Float64 tags).
            # m8n8k4:   2 D defs, 1 A + 1 B + 2 C uses.
            # m16n8k8:  4 D defs, 4 A + 2 B + 4 C uses.
            # m16n8k16: 4 D defs, 8 A + 4 B + 4 C uses.
            "DMMA.8x8x4": [[FLOAT64] * 2, [FLOAT64] * 4],
            "DMMA.16x8x8": [[FLOAT64] * 4, [FLOAT64] * 10],
            "DMMA.16x8x16": [[FLOAT64] * 4, [FLOAT64] * 16],

            # ── Missing warp ops ─────────────────────────────────────
            "REDUX": [[INT32], [INT32]],
            "REDUX.SUM": [[INT32], [INT32]],
            "REDUX.MIN": [[INT32], [INT32]],
            "REDUX.MAX": [[INT32], [INT32]],

            # ── Missing bit manipulation ─────────────────────────────
            "BFE": [[INT32], [INT32, INT32, INT32]],
            "BFI": [[INT32], [INT32, INT32, INT32, INT32]],
            "UBREV": [[INT32], [INT32]],
            "UFLO": [[INT32], [INT32]],
            "UPOPC": [[INT32], [INT32]],
            "UPRMT": [[INT32], [INT32, INT32, INT32]],

            # ── Missing shift variants ───────────────────────────────
            "USHL": [[INT32], [INT32, INT32]],
            "USHR": [[INT32], [INT32, INT32]],

            # ── Missing logic variants ───────────────────────────────
            "LOP": [[INT32], [INT32, INT32]],
            "ULOP": [[INT32], [INT32, INT32]],
            "ULOP32I": [[INT32], [INT32, INT32]],

            # ── Missing predicate ops ────────────────────────────────
            "PSETP": [[BOOL, BOOL], [BOOL, BOOL, BOOL]],
            "UPSETP64": [[BOOL, BOOL], [BOOL, BOOL, BOOL]],

            # ── Missing data movement ────────────────────────────────
            "UP2UR": [[INT32], [BOOL]],
            "CS2R": [[INT32], []],
            "CALLEXT": [[INT32], [INT64, INT64]],
            "LEPC": [[INT32], []],

            # ── Missing control flow ─────────────────────────────────
            "SWITCH": [[], [INT32]],
            "PHI": [[TOP], [TOP]],
            "PHI64": [[NUM64], [NUM64]],
            "BREAK": [[], []],
            "BRK": [[], []],
            "PBK": [[], []],
            "BSSY": [[], []],
            "BSYNC": [[], []],

            # ── Missing NOP/hint opcodes ─────────────────────────────
            "BMOV": [[], []],
            "YIELD": [[], []],
            "ERRBAR": [[], []],
            "LDGDEPBAR": [[], []],
            "MEMBAR": [[], []],
            "ENDCOLLECTIVE": [[], []],
            "CGAERRBAR": [[], []],
            # TMA hint / fence / cluster-barrier opcodes. All NOP-safe for
            # single-thread x86:
            # UTMAPF / UTMACCTL are cache/L2 prefetch hints; UTMACMDFLUSH
            # is a descriptor-cache flush (synchronous on x86); UCGABAR_*
            # are cluster barriers (single-CTA → no-op); FENCE.VIEW.ASYNC
            # ordering is trivially satisfied.
            "UTMAPF": [[], []],
            "UTMACCTL": [[], []],
            "UTMACMDFLUSH": [[], []],
            "UCGABAR_ARV": [[], []],
            "UCGABAR_WAIT": [[], []],
            "CLUSTERDIM": [[INT32], []],
            "TMEMFREE2": [[], [INT32, INT32]],
            "TMEMALLOC1": [[], [INT32, INT32]],
            "TMEMALLOC2": [[], [INT32, INT32]],
            "UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN": [[BOOL, INT32], [INT32]],
            "UGETNEXTWORKID.SELFCAST": [[], [INT32, INT32]],
            "UGETNEXTWORKID.BROADCAST": [[], [INT32, INT32]],
            "UTCCP": [[], [INT32, INT64]],
            "UTCBAR": [[], [INT32, INT32]],
            "UTCBAR.MULTICAST": [[], [INT32, INT32, INT32]],
            "UTCATOMSWS.FIND_AND_SET": [[BOOL, INT32], [INT32]],
            "UTCATOMSWS.AND": [[INT32], [INT32]],
            # TMA data-move opcodes — lift_utmaldg / lift_utmastg decompose
            # the memory effect; types of the UR operands are driven by
            # downstream propagation (descriptor reg = INT64 pointer, coord
            # = INT32 offset). Empty seed avoids "Unknown opcode" warning
            # without over-constraining the lattice for .2D/.3D/.4D/.5D.
            "UTMALDG": [[], []],
            "UTMASTG": [[], []],
            # UBLKRED.G.S.ADD.F32 [gmem_addr], [smem_off], size, desc — defines
            # no register (pure memory reduce). Empty seed; UR operand types are
            # driven by downstream propagation (gmem addr pair = INT64, smem off
            # = INT32, desc pair = INT64).
            "UBLKRED": [[], []],
        }
        
        self.propagate_table = {
            "S2R": [["A"], ["A"]],
            "MOV": [["A"], ["A"]],
            "AND": [["A"], ["A", "A"]],
            "OR": [["A"], ["A", "A"]],
            "XOR": [["A"], ["A", "A"]],
            "NOT": [["A"], ["A", "A"]],
            "SHL": [["A"], ["A", "A"]],
            "SHR": [["A"], ["A", "A"]],
            "MOVM": [["A"], ["A"]],
            "R2UR": [["A"], ["A"]],
            "PRMT": [["B"], ["A", "A", "C"]],
            
            "PACK64": [["B"], ["A", "A"]],
            "UNPACK64": [["A", "A"], ["B"]],
            "MOV64": [["A"], ["A"]],
        }

        # instruction_type_table is immutable after construction, so pre-split
        # the keys once (base opcode -> [(mods_tuple, key)]) instead of
        # re-splitting all ~280 keys on every _find_type_table call.
        self._type_table_by_base = {}
        for key in self.instruction_type_table:
            parts = key.split('.')
            self._type_table_by_base.setdefault(parts[0], []).append(
                (parts[1:], key))
        self._type_table_memo = {}


    def apply(self, module):
        super().apply(module)
        print("=== Start of TypeAnalysis ===")
        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.process_func(func)
        print("=== End of TypeAnalysis ===")


    def process_func(self, function):
        self.conflicts = set()
        # Ambiguous type resolution counters
        self.type_stats = {
            "unique_resolved": 0,      # |T(v)| == 1
            "ambiguous_int32": 0,       # |T(v)| > 1, resolved to Int32
            "ambiguous_int64": 0,       # |T(v)| > 1, resolved to Int64
            "ambiguous_other": 0,       # |T(v)| > 1, resolved to other type
            "empty_default": 0,         # |T(v)| == 0, defaulted to UInt32
            "reg_group": 0,             # resolved via reg_group path
            "total": 0,
        }
        self.type_detail_counts = {}    # track resolved type -> count
        work_list = self.traverse_cfg(function)

        optypes = {}
        
        # Static type resolution
        for bb in work_list:
            for inst in bb.instructions:
                self.resolve_types(inst, optypes)

        changed = True
        iteration = 0
        max_iters = self.max_iterations if self.max_iterations is not None else 5

        while changed and iteration < max_iters:
            changed = False

            for bb in work_list:
                changed |= self.process_bb(bb, optypes)

            iteration += 1
            if iteration >= max_iters and changed:
                print(f"Warning: TypeAnalysis stopped at {iteration} iterations"
                      f" (max={max_iters})")
                break

        # Cost-evaluation hook: record the number of fixpoint iterations this
        # function needed so a harness can correlate propagation cost with
        # kernel size. No-op unless RECORD_ITERS is flipped on (default off,
        # so the production pipeline is unaffected).
        self.last_n_iterations = iteration
        if TypeAnalysis.RECORD_ITERS:
            TypeAnalysis.ITERATION_LOG.append((function.name, iteration))

        # Apply types to instructions
        for bb in work_list:
            for inst in bb.instructions:
                for op in inst.operands:
                    if op.is_reg_group:
                        for child_op in op.reg_group:
                            child_op.type_desc = self.get_type_desc(child_op, optypes)
                    op.type_desc = self.get_type_desc(op, optypes)
                        

        # Store optypes for export
        function._typeanalysis_optypes = optypes
        function._typeanalysis_conflicts = set(self.conflicts)

        # Print ambiguous type resolution statistics
        s = self.type_stats
        total_scalar = s["total"] - s["reg_group"]
        ambiguous_total = s["ambiguous_int32"] + s["ambiguous_int64"] + s["ambiguous_other"]
        print(f"\n{'='*60}")
        print(f"TYPE RESOLUTION STATS for {function.name}")
        print(f"{'='*60}")
        print(f"  Total operands resolved:        {s['total']}")
        print(f"    Scalar operands:              {total_scalar}")
        print(f"    Reg-group operands:           {s['reg_group']}")
        print(f"  --- Scalar breakdown ---")
        print(f"    Uniquely resolved (|T|=1):    {s['unique_resolved']}")
        print(f"    Ambiguous (|T|>1):            {ambiguous_total}")
        if ambiguous_total > 0:
            print(f"      -> Int32 fallback:          {s['ambiguous_int32']}")
            print(f"      -> Int64 fallback:          {s['ambiguous_int64']}")
            print(f"      -> Other type:              {s['ambiguous_other']}")
        print(f"    Empty set (|T|=0, UInt32):    {s['empty_default']}")
        if total_scalar > 0:
            pct_unique = 100.0 * s["unique_resolved"] / total_scalar
            pct_ambig = 100.0 * ambiguous_total / total_scalar
            pct_empty = 100.0 * s["empty_default"] / total_scalar
            print(f"  --- Percentages (scalar) ---")
            print(f"    Unique:  {pct_unique:.2f}%")
            print(f"    Ambiguous: {pct_ambig:.2f}%")
            print(f"    Empty:   {pct_empty:.2f}%")
        print(f"  --- Resolved type distribution ---")
        for t, c in sorted(self.type_detail_counts.items(), key=lambda x: -x[1]):
            print(f"    {t:20s}: {c}")
        print(f"{'='*60}\n")

        print("done")
        print("=== End of TypeAnalysis ===")
        
    def get_type_desc(self, operand, optypes):
        
        if operand.is_reg_group:
            self.type_stats["total"] += 1
            self.type_stats["reg_group"] += 1
            reg_count = len(operand.reg_group)
            reg_types = [self.get_optype(optypes, sub_op) for sub_op in operand.reg_group]
            common_types = reg_types[0]
            for rt in reg_types[1:]:
                common_types = common_types & rt
            common_types = common_types & NUM32

            if len(common_types) == 0:
                common_types = reg_types[0] & NUM32 if len(reg_types[0] & NUM32) > 0 else NUM32

            if reg_count == 2:
                types = self.type_32_to_64(common_types)
                if len(INT64 & types) > 0:
                    resolved = list(INT64)[0]
                elif len(FLOAT64 & types) > 0:
                    resolved = list(FLOAT64)[0]
                else:
                    resolved = list(types)[0]
            elif reg_count == 4:
                types = self.type_32_to_128(common_types)
                if len(INT128 & types) > 0:
                    resolved = list(INT128)[0]
                else:
                    resolved = list(types)[0]
            else:
                resolved = f"Int{reg_count * 32}"
            self.type_detail_counts[resolved] = self.type_detail_counts.get(resolved, 0) + 1
            return resolved

        elif operand in optypes:
            types = optypes[operand]
        elif operand.reg in optypes:
            types = optypes[operand.reg]
        else:
            raise ValueError(f"Type not found for operand {operand} / {operand.reg}")
        
        self.type_stats["total"] += 1

        if len(types) == 1:
            self.type_stats["unique_resolved"] += 1
        elif len(types) > 1:
            print(f"Warning: Multiple possible types for {operand} / {operand.reg}: {types}")

        if len(INT32 & types) > 0:
            resolved = list(INT32)[0]
            if len(types) > 1:
                self.type_stats["ambiguous_int32"] += 1
        elif len(INT64 & types) > 0:
            resolved = list(INT64)[0]
            if len(types) > 1:
                self.type_stats["ambiguous_int64"] += 1
        elif len(FLOAT32 & types) > 0 and len(TF32 & types) > 0:
            # TF32 storage is a 32-bit float with 13 zeroed mantissa bits.
            # When a register flows between a memory load (Float32 tag)
            # and a tensor-core use (TF32 tag), treat it as Float32 so
            # the IR type stays stable — TF32 only matters at the HMMA
            # seed point, not in the general lattice.
            resolved = list(FLOAT32)[0]
            self.type_stats["ambiguous_other"] += 1
        elif len(types) > 0:
            resolved = list(types)[0]
            if len(types) > 1:
                self.type_stats["ambiguous_other"] += 1
        else:
            # Empty-set fallback. Under CULIFTER_NO_FALLBACK (used by the B3/B4
            # ablation harness), refuse to invent a type — the point of those
            # ablations is to observe lift failure when type recovery is
            # incomplete. Without this, the Int32 fallback silently saves
            # virtually every kernel even when propagation is disabled.
            import os as _os
            if _os.environ.get("CULIFTER_NO_FALLBACK") == "1":
                raise Exception(
                    f"TypeAnalysis: no type for operand (strict mode, "
                    f"CULIFTER_NO_FALLBACK=1)"
                )
            resolved = DEFAULT_UNTYPED
            self.type_stats["empty_default"] += 1

        self.type_detail_counts[resolved] = self.type_detail_counts.get(resolved, 0) + 1
        return resolved
        
    def _is_subseq(self, key_mods, inst_mods):
        j = 0
        for mod in inst_mods:
            if j < len(key_mods) and mod == key_mods[j]:
                j += 1
        return j == len(key_mods)

    def _find_type_table(self, opcodes):
        memo_key = tuple(opcodes)
        try:
            return self._type_table_memo[memo_key]
        except KeyError:
            pass

        base, inst_mods = opcodes[0], opcodes[1:]
        best_key, best_len = None, -1

        for key_mods, key in self._type_table_by_base.get(base, ()):
            if self._is_subseq(key_mods, inst_mods) and len(key_mods) > best_len:
                best_key, best_len = key, len(key_mods)

        result = self.instruction_type_table.get(best_key)
        self._type_table_memo[memo_key] = result
        return result

    def resolve_types(self, inst, optypes):
        opcode = inst.opcodes[0]

        # Optional ablation hook: pretend the listed opcodes don't seed so
        # callers can measure progressive seed degradation. The blacklist is
        # comma-separated, e.g.
        #   SLIFTER_SEED_BLACKLIST=FADD,IADD3,LDG,ISETP
        if self._seed_blacklist and opcode in self._seed_blacklist:
            for op in inst.operands:
                self.set_op_type(optypes, op, TOP, inst)
            return

        if opcode in inst.TCGEN_MMA:
            for op in inst.get_uses():
                ty = BOOL if op.is_predicate_reg or op.is_pt else INT64 if op.is_reg_group else INT32
                self.set_op_type(optypes, op, ty, inst)
            return

        if opcode in ("LDTM", "STTM") and inst.tmem_transfer:
            for operand in inst.get_defs():
                self.set_op_type(optypes, operand, NUM32, inst)
            for operand in inst.get_uses():
                self.set_op_type(optypes, operand, INT32, inst)
            return

        # Phi, switch(pseudo-inst) can have any number of operands, all same type
        if opcode == "PHI" or opcode == "PHI64" or opcode == "SWITCH":
            if opcode == "PHI64":
                inst.opcodes[0] = "PHI"
            set_op_type = self.set_op_type
            phi_type = TOP
            for op in inst.operands:
                set_op_type(optypes, op, phi_type, inst)
            return
        
        # HGMMA/WGMMA/QGMMA: dynamic type table based on actual operand count
        # (the static table may not match the actual register count for large shapes)
        if opcode in ("HGMMA", "WGMMA", "QGMMA", "IGMMA"):
            defs = inst.get_defs()
            uses = inst.get_uses()
            _, accum = inst.wgmma_shape_accum()
            if accum == "S32" or (inst.opcodes[1:2] == ["SP"] and accum == "F16"):
                packed_type = INT32
            elif accum.startswith("F16"):
                packed_type = FLOAT16
            else:
                packed_type = FLOAT32
            for def_op in defs:
                self.set_op_type(optypes, def_op, packed_type, inst)
            for use_op in uses:
                typ = INT32 if getattr(use_op, "wgmma_sparse_meta", False) else packed_type
                self.set_op_type(optypes, use_op, typ, inst)
            return

        curr_type_table = self._find_type_table(inst.opcodes)

        if not curr_type_table:
            print(f"Warning: Unknown opcode {inst.opcodes[0]} in StaticResolve")
            for op in inst.operands:
                self.set_op_type(optypes, op, TOP, inst)
            return
        
        optype_map = {}
    
        for i, def_op in enumerate(inst.get_defs()):
            if i < len(curr_type_table[0]) and curr_type_table[0][i] != "PROP":
                optype_map[def_op] = curr_type_table[0][i]
            else:
                optype_map[def_op] = TOP
                
        for i, use_op in enumerate(inst.get_uses()):
            if i < len(curr_type_table[1]) and curr_type_table[1][i] != "PROP":
                optype_map[use_op] = curr_type_table[1][i]
            else:
                optype_map[use_op] = TOP
                        
        # Apply types
        for operand, type_desc in optype_map.items():
            if operand.is_predicate_reg or operand.is_pt:
                self.set_op_type(optypes, operand, BOOL, inst)
            else:
                self.set_op_type(optypes, operand, type_desc, inst)
            

    def get_optype(self, optypes, operand):
        
        if operand.is_reg_group:
            reg_types = [self.get_optype(optypes, sub_op) for sub_op in operand.reg_group]
            common_types = reg_types[0]
            for rt in reg_types[1:]:
                common_types = common_types & rt
            common_types = common_types & NUM32

            if len(common_types) == 0:
                print(f"Warning: Reg group {operand} has type mismatch: {reg_types}")
                for sub_op in operand.reg_group:
                    self.conflicts.add(sub_op.reg)
                common_types = reg_types[0] & NUM32 if len(reg_types[0] & NUM32) > 0 else NUM32

            if len(operand.reg_group) == 2:
                return self.type_32_to_64(common_types)
            elif len(operand.reg_group) == 4:
                return self.type_32_to_128(common_types)
            else:
                return common_types

        if operand in optypes:
            return optypes[operand]
        elif operand.reg in optypes:
            return optypes[operand.reg]
        else:
            return TOP

    @staticmethod
    def _store_op_type(optypes, key, value):
        """Write optypes[key] = value; report whether the entry changed."""
        prev = optypes.get(key, _MISSING)
        if prev is not _MISSING and prev == value:
            return False
        optypes[key] = value
        return True

    def set_op_type(self, optypes, operand, type_set, inst):
        """Returns True if any optypes entry changed (for fixed-point check)."""
        previous_type = self.get_optype(optypes, operand)
        changed = False

        # Record type conflict
        if len(previous_type & type_set) == 0:
            if operand.reg not in self.conflicts:
                print(f"Warning: Type mismatch for {operand} in {inst}: prevType: {previous_type} vs optype: {type_set}")
                self.conflicts.add(operand.reg)
            changed |= self._store_op_type(optypes, operand, type_set)

            if operand.is_reg_group:
                if operand.is_reg_pair:
                    base_type = self.type_64_to_32(type_set)
                elif operand.is_reg_quad:
                    base_type = self.type_128_to_32(type_set)
                else:
                    raise ValueError("Not possible")
                for sub_op in operand.reg_group:
                    self.conflicts.add(sub_op.reg)
                    changed |= self._store_op_type(optypes, sub_op, base_type)
            return changed

        if operand.is_reg_group:
            if operand.is_reg_pair:
                base_type = self.type_64_to_32(type_set)
            elif operand.is_reg_quad:
                base_type = self.type_128_to_32(type_set)
            else:
                raise ValueError("Not possible")
            for sub_op in operand.reg_group:
                changed |= self.set_op_type(optypes, sub_op, base_type, inst)
            return changed

        if operand.is_writable_reg:
            if operand.reg not in self.conflicts:
                changed |= self._store_op_type(
                    optypes, operand.reg, previous_type & type_set)
            else:
                changed |= self._store_op_type(optypes, operand, type_set)
        else:
            # Store operand itself as key for non-register values
            # E.g. 0 in IADD vs 0 in FADD have different types
            changed |= self._store_op_type(
                optypes, operand, previous_type & type_set)
        return changed

    def type_64_to_32(self, type_set):
        table = {
            "Int64": "Int32",
            "Float64": "Float32"
        }
        new_type_set = set()
        for t in type_set:
            if t in table:
                new_type_set.add(table[t])
        return new_type_set

    def type_32_to_64(self, type_set):
        table = {
            "Int32": "Int64",
            "Float32": "Float64",
        }
        new_type_set = set()
        for t in type_set:
            if t in table:
                new_type_set.add(table[t])
        return new_type_set

    def type_32_to_128(self, type_set):
        table = {
            "Int32": "Int128",
            "Float32": "Float128",
        }
        new_type_set = set()
        for t in type_set:
            if t in table:
                new_type_set.add(table[t])

        if len(new_type_set) == 0:
            # Fallback for type sets without 32-bit types (e.g. TOP with only Bool).
            new_type_set = {"Int128"}
        return new_type_set

    def type_128_to_32(self, type_set):
        table = {
            "Float128": "Float32",
            "Int128": "Int32",
        }
        new_type_set = set()
        for t in type_set:
            if t in table:
                new_type_set.add(table[t])

        if len(new_type_set) == 0:
            # Fallback: type_set has no 128-bit types (e.g. TOP from unknown opcode).
            # Map 32-bit types directly as the base element type.
            for t in type_set:
                if t in ("Int32", "Float32"):
                    new_type_set.add(t)
            if len(new_type_set) == 0:
                new_type_set = {"Int32"}
        return new_type_set

    def traverse_cfg(self, function):
        entry_bb = function.blocks[0]
        visited = set()
        queue = deque([entry_bb])
        work_list = []

        while queue:
            curr_bb = queue.popleft()
            
            if curr_bb in visited:
                continue

            visited.add(curr_bb)
            work_list.append(curr_bb)

            for succ_bb in curr_bb.succs:
                if succ_bb not in visited:
                    queue.append(succ_bb)

        return work_list

    def process_bb(self, bb, optypes):
        changed = False
        for inst in bb.instructions:
            changed |= self.propagate_types(inst, optypes)
        return changed

    def propagate_types(self, inst, optypes):
        opcode = inst.opcodes[0]
        changed = False

        # Phi can have any number of operands, all same type
        if opcode in ("PHI", "PHI64"):
            all_ops = list(inst.get_defs()) + list(inst.get_uses())

            prop_type = TOP
            for op in all_ops:
                op_type = self.get_optype(optypes, op)
                if len(prop_type & op_type) != 0:
                    prop_type = prop_type & op_type

            for op in all_ops:
                changed |= self.set_op_type(optypes, op, prop_type, inst)
            return changed

        if opcode not in self.propagate_table:
            return False

        prop_table = self.propagate_table[opcode]

        prop_ops = {}
        for i, def_op in enumerate(inst.get_defs()):
            if i < len(prop_table[0]):
                prop_ops.setdefault(prop_table[0][i], []).append(def_op)

        for i, use_op in enumerate(inst.get_uses()):
            if i < len(prop_table[1]):
                prop_ops.setdefault(prop_table[1][i], []).append(use_op)

        for prop_key in prop_ops:
            prop_ops_list = prop_ops[prop_key]

            prop_type = TOP
            for prop_op in prop_ops_list:
                # conflict reg not participate in propagate type calculation
                op_type = self.get_optype(optypes, prop_op)

                if len(prop_type & op_type) != 0:
                    prop_type = prop_type & op_type

            for prop_op in prop_ops_list:
                changed |= self.set_op_type(optypes, prop_op, prop_type, inst)
        return changed

    # ------------------------------------------------------------------
    # Export helpers for type evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def export_type_map(func):
        """Return {register_name: inferred_type_desc} for all typed operands.

        Must be called after process_func() has run on the function.
        """
        type_map = {}
        for bb in func.blocks:
            for inst in bb.instructions:
                for op in inst.operands:
                    if op.type_desc and op.type_desc != "NOTYPE":
                        if hasattr(op, "reg") and op.reg:
                            # Keep first assignment (closest to definition)
                            if op.reg not in type_map:
                                type_map[op.reg] = op.type_desc
        return type_map

    # Const-bank value loads: the c[bank][offset] operand is the SOURCE being
    # loaded, so its own type is the address class (Int64/Int32 per the load
    # rule) while the loaded VALUE (the def) carries the recovered type. For a
    # by-value floating-point parameter (LDC.64 then bitcast at the FP use)
    # the operand reads Int but the def correctly resolves to Float. Excludes
    # LDG/LDS/LDL/LD, where a c[bank][offset] would be an *address* (pointer
    # param) and the def is dereferenced data, not the param.
    _CONST_VALUE_LOADS = frozenset({"LDC", "ULDC", "LDCU"})

    @staticmethod
    def export_const_mem_types(func):
        """Return {(bank, offset): inferred_type_desc} for kernel parameters
        read from constant memory.

        For const-bank value loads we report the loaded value's (def's)
        recovered type so by-value FP params are not mislabelled as their
        integer bit-container; value-load types take precedence over the
        operand type recorded for direct-compute uses.
        """
        const_types = {}
        val_keys = set()
        for bb in func.blocks:
            for inst in bb.instructions:
                base = inst.opcodes[0] if getattr(inst, "opcodes", None) else ""
                is_value_load = base in TypeAnalysis._CONST_VALUE_LOADS
                def_td = None
                if is_value_load:
                    for d in inst.get_defs():
                        t = getattr(d, "type_desc", None)
                        if t and t != "NOTYPE":
                            def_td = t
                            break
                for op in inst.get_uses():
                    if not getattr(op, "is_const_mem", False):
                        continue
                    bank = getattr(op, "const_mem_bank", None)
                    offset = getattr(op, "offset_value", None)
                    if bank is None or offset is None:
                        continue
                    key = (bank, offset)
                    if is_value_load and def_td:
                        if key not in val_keys:
                            const_types[key] = def_td
                            val_keys.add(key)
                    elif key not in const_types:
                        td = op.type_desc
                        if td and td != "NOTYPE":
                            const_types[key] = td

        return const_types

    @staticmethod
    def export_conflicts(func):
        """Return set of register names with type conflicts."""
        return getattr(func, "_typeanalysis_conflicts", set())
