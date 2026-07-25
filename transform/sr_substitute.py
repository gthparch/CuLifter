from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction


# Const-bank-0 special-register layout differs by architecture generation:
#   - LEGACY (Kepler/Maxwell, sm_35/sm_52): NTID.X at 0x08
#   - MODERN (Volta and newer, sm_70+):     NTID.X at 0x00
# Every supported ISA MUST appear in exactly one of these sets. A new arch
# that is left unclassified is a hard error (see the raise below) rather than
# a silent fallback: with the wrong offsets, SRSubstitute(Reverse) fails to
# recognize the launch-dim refs and the lifted blockIdx*blockDim term collapses
# to 0 (only block 0 runs) with no diagnostic. The set must track the ISAs the
# parser accepts (see parse/parser.py).
_LEGACY_ISAS = {'sm_35', 'sm_52'}
_MODERN_ISAS = {
    'sm_70', 'sm_70a', 'sm_75', 'sm_80', 'sm_80a', 'sm_86', 'sm_86a',
    'sm_89', 'sm_89a', 'sm_90', 'sm_90a', 'sm_100', 'sm_100a',
    'sm_120', 'sm_120a',
}


def _sr_map_for_isa(isa: str | None):
    # Legacy layout (sm_35/sm_52): NTID.X at 0x08
    base = {
        'SR_NTID.X': 0x08,
        'SR_NTID.Y': 0x0C,
        'SR_NTID.Z': 0x10,
        'SR_GRID_DIM.X': 0x14,
        'SR_GRID_DIM.Y': 0x18,
        'SR_GRID_DIM.Z': 0x1C,
        'SR_CTAID.X': 0x20,
        'SR_CTAID.Y': 0x24,
        'SR_CTAID.Z': 0x28,
        'SR_TID.X': 0x2C,
        'SR_TID.Y': 0x30,
        'SR_TID.Z': 0x34,
        'SR_LANEID': 0x38,
    }
    if isa in _LEGACY_ISAS:
        return base
    if isa not in _MODERN_ISAS:
        raise ValueError(
            f"_sr_map_for_isa: unclassified ISA {isa!r}. Add it to "
            f"_LEGACY_ISAS or _MODERN_ISAS in transform/sr_substitute.py after "
            f"confirming its const-bank-0 dim layout (the NTID.X offset). Do "
            f"not add a silent default — the wrong offsets corrupt blockDim."
        )
    # Modern layout (Volta+): launch-dim builtins at the start of bank 0.
    base = dict(base)
    base['SR_NTID.X'] = 0x0
    base['SR_NTID.Y'] = 0x4
    base['SR_NTID.Z'] = 0x8
    base['SR_GRID_DIM.X'] = 0xC
    base['SR_GRID_DIM.Y'] = 0x10
    base['SR_GRID_DIM.Z'] = 0x14
    if isa in ('sm_90', 'sm_90a'):
        base = dict(base)
        # BUG-067: On SM90, c[0][0x28] is the LOCAL MEMORY FRAME POINTER
        # (every kernel starts with LDC R1, c[0][0x28]), NOT SR_CTAID.Z.
        # Native-H100 verification confirmed this.
        # Remove the mapping so that:
        #   1. kernel_wrapper can write local_mem base to 0x28
        #   2. Any S2R SR_CTAID.Z returns constant 0 (fine for 1D/2D grids)
        del base['SR_CTAID.Z']
        # SM90 CGA CTA ID: controls per-CTA shared memory partition offset.
        # ULEA uses CgaCtaId << 24 to compute the partition base within
        # the cluster's shared memory space.
        #
        # x86 path: We execute one CTA at a time with a single 32KB
        # shared_mem buffer, so CgaCtaId MUST be 0. Mapping it to
        # CTAID.X (blockIdx) causes out-of-bounds access when blockIdx > 0.
        # Omitting from map → lift_s2r returns constant 0.
        #
        # TODO(nvvm): For the NVVM path, CgaCtaId should map to the actual
        # CTA rank within the CGA cluster. When lifting for GPU re-execution
        # with multi-CTA clusters, this needs to emit a proper intrinsic
        # (e.g. llvm.nvvm.read.ptx.sreg.clusterctarank) instead of 0.
        # See: https://docs.nvidia.com/cuda/parallel-thread-execution/
        #      #special-registers-clusterctarank
    return base


class SRSubstitute(SaSSTransform):
    def apply(self, module):
        print("=== Start of SRSubstitute ===")
        count = 0
        sr_to_offset = _sr_map_for_isa(getattr(module, 'isa', None))

        for func in module.functions:
            for block in func.blocks:
                count += process(block.instructions, sr_to_offset)

        print(f"SRSubstitute: processed {count} operands")
        print("=== End of SRSubstitute ===")


SR_OPCODES = {'S2R': 'MOV', 'S2UR': 'UMOV', 'CS2R': 'MOV'}


def process(instructions, sr_to_offset):
    count = 0

    for inst in instructions:
        if inst.opcodes and inst.opcodes[0] in SR_OPCODES:
            sr_name = inst.operands[1].name
            if sr_name not in sr_to_offset:
                continue
            offset = sr_to_offset[sr_name]

            inst.opcodes[0] = SR_OPCODES[inst.opcodes[0]]
            new_operand = Operand.from_cm(
                name=f"c[0x0][{hex(offset)}]",
                arg_offset=offset,
            )
            inst.operands[1] = new_operand
            count += 1

    return count

# Default export (legacy layout) for the CPU lifter's standalone SR lookups
# that have no module ISA in hand. _sr_map_for_isa now rejects None/unknown
# ISAs, so pin this to a known legacy arch instead of passing None.
SR_TO_OFFSET = _sr_map_for_isa('sm_52')
