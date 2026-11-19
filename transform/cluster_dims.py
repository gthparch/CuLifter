"""Recover ptxas' SM100 indexed cluster-dimension and reciprocal table loads."""
from transform.transform import SaSSTransform


class BlackwellClusterDims(SaSSTransform):
    def apply(self, module):
        if module.isa not in ("sm_100", "sm_100a"):
            return
        for func in module.functions:
            for block in func.blocks:
                for load in block.instructions:
                    self.recover(load)

    @staticmethod
    def recover(load):
        if load.opcodes not in (["LDC"], ["LDCU"], ["ULDC"]) or len(load.operands) != 2:
            return
        mem = load.operands[1]
        if not mem.is_const_mem or mem.const_mem_bank != 0 or not mem.reg:
            return
        offsets = (0x2a0, 0x2a4, 0x2a8, 0x2b0, 0x2b4, 0x2b8)
        if mem.offset not in offsets:
            return
        # A phi or a predicated definition is insufficient. The table selector
        # must come from this exact physical register, optionally through R2UR.
        scale = load.reaching_defs.get(mem)
        if scale and scale.opcodes == ["R2UR"] and not scale.pflag:
            scale = scale.reaching_defs.get(scale.operands[1])
        if not scale or scale.opcodes != ["IMAD"] or scale.pflag or len(scale.operands) != 4:
            return
        _, idx, factor, addend = scale.operands
        if not factor.is_immediate or (factor.immediate_value & 0xffffffff) != 0xffffff60 or not addend.is_rz:
            return
        selector = scale.reaching_defs.get(idx)
        if not selector or selector.opcodes != ["CS2R", "32"] or selector.pflag or selector.operands[1].name != "SR_CgaSize":
            return
        field = offsets.index(mem.offset)
        load.opcodes = ["CLUSTERDIM", "RECIP" if field >= 3 else "COUNT", "XYZ"[field % 3]]
        load.operands = [load.operands[0]]
