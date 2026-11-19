from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

from transform.sr_substitute import _sr_map_for_isa


PRE_HOPPER_STACK_POINTER_ISAS = frozenset({
    "sm_70", "sm_70a", "sm_75", "sm_80", "sm_80a", "sm_86", "sm_86a", "sm_89", "sm_89a",
})


class SRSubstituteReverse(SaSSTransform):
    def apply(self, module):
        print("=== Start of SRSubstituteReverse ===")
        count = 0

        isa = module.isa
        offset_to_sr = {offset: sr for sr, offset in _sr_map_for_isa(isa).items()}
        if isa in PRE_HOPPER_STACK_POINTER_ISAS:
            # Direct constant-bank operands use the stack-pointer slot.
            # Explicit S2R instructions still read the real block index.
            del offset_to_sr[0x28]

        for func in module.functions:
            for block in func.blocks:
                count += process(block.instructions, offset_to_sr)

        self.total_sr_reverse = count
        print(f"SRSubstituteReverse: processed {count} operands")
        print("=== End of SRSubstituteReverse ===")


def process(instructions, offset_to_sr):
    count = 0

    # Track which offsets have been processed and their corresponding temp register names
    offset_to_temp_reg = {}
    new_instructions = []

    for inst in instructions:
        for operand in inst.operands:
            if operand.is_arg and operand.offset in offset_to_sr:
                offset = operand.offset

                if offset not in offset_to_temp_reg:
                    sr_name = offset_to_sr[offset]
                    temp_reg_name = f"sr_temp_{hex(offset)}"
                    offset_to_temp_reg[offset] = temp_reg_name

                    sr_operand = Operand.from_reg(sr_name, sr_name)
                    temp_operand = Operand.from_reg(temp_reg_name, temp_reg_name)

                    s2r_inst = Instruction(
                        id=f"s2r_{hex(offset)}",
                        opcodes=["S2R"],
                        operands=[temp_operand, sr_operand],
                        parentBB=inst.parent,
                    )

                    new_instructions.append(s2r_inst)
                    count += 1

        for idx, operand in enumerate(inst.operands):
            if operand.is_arg and operand.offset in offset_to_sr:
                offset = operand.offset
                temp_reg_name = offset_to_temp_reg[offset]

                new_operand = Operand.from_reg(
                    temp_reg_name,
                    temp_reg_name,
                    suffix=operand.suffix,
                    reuse=operand.reuse,
                )
                new_operand.set_type_desc(operand.get_type_desc())
                inst.operands[idx] = new_operand

        new_instructions.append(inst)

    instructions.clear()
    instructions.extend(new_instructions)

    return count
