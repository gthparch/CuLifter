from transform.transform import SaSSTransform

class MovEliminate(SaSSTransform):
    def _is_plain_register_source(self, op):
        if not op or not op.is_reg:
            return False
        if op.is_reg_group or op.is_mem_addr or op.is_const_mem or op.is_immediate:
            return False
        # Only eliminate true identity moves. Any modifier changes semantics.
        if op.is_negative_reg or op.is_not_reg or op.is_abs_reg:
            return False
        if op.suffix is not None:
            return False
        return True

    def apply(self, module):
        print("=== Start of MovEliminate ===")
        count = 0

        for func in module.functions:
            print(f"Processing function: {func.name}")
            for block in func.blocks:
                count += self.process_block(block)
                

        print(f"MovEliminate: removed {count} mov instructions")
        print("=== End of MovEliminate ===")

    def process_block(self, block):
        count = 0
        new_instructions = []
        
        
        for inst in block.instructions:
            if self.checkInstruction(inst):
                new_instructions.append(inst)
            else:
                print(f"Removed: {inst}")
                count += 1
        
        block.instructions = new_instructions
        return count
    
    
    def convert_mov(self, inst):
        
        # UMOV UR4, UR7 => MOV UR4, UR7
        if inst.opcodes and inst.opcodes[0] == 'UMOV' and len(inst.operands) >= 2:
            dst_op = inst.operands[0].clone()
            src_op = inst.operands[1].clone()
            new_inst = inst.clone()
            new_inst.opcodes = ['MOV']
            new_inst.operands = [dst_op, src_op]
            return new_inst
        
        # IMAD.MOV R13, RZ, RZ, -R13 => MOV R13, -R13
        if (inst.opcodes and len(inst.opcodes) >= 2 and inst.opcodes[0] == 'IMAD' and inst.opcodes[1] == "MOV"):
            if len(inst.operands) >= 4:
                op1 = inst.operands[1]
                op2 = inst.operands[2]
                if op1.is_rz and op2.is_rz:
                    dst_op = inst.operands[0].clone()
                    src_op = inst.operands[3].clone()
                    new_inst = inst.clone()
                    new_inst.opcodes = ['MOV']
                    new_inst.operands = [dst_op, src_op]
                    return new_inst
                
    def checkInstruction(self, inst):
        
        status = True
        
        # MOV R17, R58
        if (
            inst.opcodes
            and inst.opcodes[0] == 'MOV'
            and len(inst.operands) >= 2
            and self._is_plain_register_source(inst.operands[1])
        ):
            status = False
            reg = inst.operands[1]
        
        # UMOV UR4, UR7
        if (
            inst.opcodes
            and inst.opcodes[0] == 'UMOV'
            and len(inst.operands) >= 2
            and self._is_plain_register_source(inst.operands[1])
        ):
            status = False
            reg = inst.operands[1]
        
        # IMAD.MOV R13, RZ, RZ, -R13
        if (inst.opcodes and len(inst.opcodes) >= 2 and inst.opcodes[0] == 'IMAD' and inst.opcodes[1] == "MOV"):
            if len(inst.operands) >= 4:
                op1 = inst.operands[1]
                op2 = inst.operands[2]
                if op1.is_rz and op2.is_rz and self._is_plain_register_source(inst.operands[3]):
                    status = False
                    reg = inst.operands[3]
            
        # Rename        
        if not status:
            for _, userOp in inst.users.get(inst.operands[0], set()):
                if userOp.reg != inst.operands[0].reg:
                    raise Exception("Should not happen")
                userOp.set_reg(reg.reg)

            return False

        return True
