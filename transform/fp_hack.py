from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class FPHack(SaSSTransform):
    def apply(self, module):
        print("=== Start of FPHack ===")
        count = 0

        for func in module.functions:
            count += self.process(func)

        print(f"FPHack: replaced {count} pattern1")
        print("=== End of FPHack ===")

    # Fast integer division
    # I2F.F32.U32.RP R4, R0 ;
    # MUFU.RCP R4, R4 ;
    # IADD32I R5, R4, 0xffffffe ;  or IADD3 R5, R4, 0xffffffe, RZ
    # F2I.FTZ.U32.F32.TRUNC R5, R5 ; 
    # This hack produces a magic number for fast integer division
    
    # to make type system working, insert BITCAST before and after IADD32I
    def handle_pattern1(self, inst, insert_insts, remove_insts):
        if inst.opcodes[0] != "I2F":
            return 0
        
        inst1 = inst
        inst2 = None
        inst3 = None
        inst4 = None

        for users in inst.users.values():
            for use_inst, use_op in users:
                if use_inst.opcodes[0] == "MUFU":
                    inst2 = use_inst

        if not inst2:
            return 0
        
        for users in inst2.users.values():
            for use_inst, use_op in users:
                if use_inst.opcodes[0] == "IADD32I" or use_inst.opcodes[0] == "IADD3":
                    inst3 = use_inst
                    iadd_use_op = use_op

        if not inst3:
            return 0
        
        for users in inst3.users.values():
            for use_inst, use_op in users:
                if use_inst.opcodes[0] == "F2I":
                    inst4 = use_inst
                    f2i_use_op = use_op

        if not inst4:
            return 0

        src_op = inst2.get_defs()[0].clone()
        dest_op_name = src_op.name + "_cast"    
        dest_op = Operand.from_reg(dest_op_name, dest_op_name)
        iadd_use_op.set_reg(dest_op_name)

        for users in inst2.users.values():
            for use_inst, use_op in users:
                use_op.set_reg(dest_op_name)

        bitcast_inst_before = Instruction(
            id=f"{inst2.id}_bitcast_before",
            opcodes=["BITCAST"],
            operands=[dest_op, src_op],
            parentBB=inst2.parent
        )
        insert_insts[inst2] = bitcast_inst_before
        
        src_op = inst3.get_defs()[0].clone()
        dest_op_name = src_op.name + "_cast"    
        dest_op = Operand.from_reg(dest_op_name, dest_op_name)
        f2i_use_op.set_reg(dest_op_name)

        bitcast_inst_after = Instruction(
            id=f"{inst3.id}_bitcast_after",
            opcodes=["BITCAST"],
            operands=[dest_op, src_op],
            parentBB=inst3.parent
        )
        insert_insts[inst3] = bitcast_inst_after

        return 1


    def process(self, func):

        insert_insts = {}
        remove_insts = set()

        count = 0

        for block in func.blocks:
            for inst in block.instructions:
                count += self.handle_pattern1(inst, insert_insts, remove_insts)

        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                if inst not in remove_insts:
                    new_instructions.append(inst)
                if inst in insert_insts:
                    new_instructions.append(insert_insts[inst])

            block.instructions = new_instructions
        
        return count
