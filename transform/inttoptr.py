from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class IntToPtr(SaSSTransform):
    def apply(self, module):
        print("=== Start of IntToPtr Transformation ===")
        count = 0
        new_regs_count = {}
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    if inst.is_global_load() or inst.is_global_store():
                        ptr_idx = inst.use_op_start_idx

                        src_op = inst.get_uses()[0].clone()

                        new_reg = src_op.reg + "_to_ptr"
                        new_regs_count[new_reg] = new_regs_count.get(new_reg, 0) + 1
                        new_reg = f"{new_reg}_{new_regs_count[new_reg]}"
                        dst_op = Operand.from_reg(new_reg, new_reg)

                        cast_inst = Instruction(
                            id=f"{inst.id}_inttoptr",
                            opcodes=["INTTOPTR"],
                            operands=[dst_op, src_op],
                            parentBB=inst.parent
                        )
                        new_insts.append(cast_inst)
                        count += 1

                        inst.get_uses()[0].set_reg(new_reg)
                        
                    new_insts.append(inst)
                block.instructions = new_insts

        print(f"Total INTTOPTR instructions added: {count}")
        print(f"=== End of IntToPtr Transformation ===")
