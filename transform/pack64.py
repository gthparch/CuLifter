from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class Pack64(SaSSTransform):

    def _next_hi_name(self, reg_name):
        if reg_name.startswith('UR'):
            prefix = 'UR'
            rest = reg_name[2:]
        else:
            prefix = 'R'
            rest = reg_name[1:]
        return f"{prefix}{int(rest)+1}"

    def _prev_hi_name(self, reg_name):
        if reg_name.startswith('UR'):
            prefix = 'UR'
            rest = reg_name[2:]
        else:
            prefix = 'R'
            rest = reg_name[1:]
        return f"{prefix}{int(rest)-1}"

    def _is_simple_hw_reg(self, reg_name):
        if reg_name.startswith('UR'):
            rest = reg_name[2:]
        elif reg_name.startswith('R'):
            rest = reg_name[1:]
        else:
            return False
        return rest.isdigit()

    def create_pack64(self, addr_op, parentBB):
        src_op_lower = Operand.from_reg(addr_op.reg, addr_op.reg)

        src_op_upper_name = self._next_hi_name(addr_op.reg)
        src_op_upper = Operand.from_reg(src_op_upper_name, src_op_upper_name)

        dest_op_name = src_op_lower.reg + "_int64"
        dst_op = Operand.from_reg(dest_op_name, dest_op_name)

        pack64_inst = Instruction(
            id=f"pack64_{addr_op.name}",
            opcodes=["PACK64"],
            operands=[
                dst_op,
                src_op_lower,
                src_op_upper
            ],
            parentBB=parentBB
        )

        addr_op.set_reg(dest_op_name)

        return pack64_inst

    def create_iadd64(self, addr_op, offset_op, inst):
        src_op_base = Operand.from_reg(addr_op.name, addr_op.name)
        src_op_offset = Operand.from_reg(offset_op.name, offset_op.name)
        dest_op_name = f"{addr_op.name}_iadd64"
        dest_op = Operand.from_reg(dest_op_name, dest_op_name)

        iadd64_inst = Instruction(
            id=f"iadd64_{addr_op.name}",
            opcodes=["IADD64"],
            operands=[
                dest_op,
                src_op_base,
                src_op_offset
            ],
            parentBB=inst.parent
        )

        addr_op.replace(dest_op)

        return iadd64_inst

    def apply(self, module):
        print("=== Start of Pack64 Transformation ===")


        count = 0
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                #     for op in inst.operands:
                #         if not op.is_mem_addr:
                #             continue

                #         if op.is_rz:
                #             continue

                #         ur_offset = True if op.mem_addr_offset and "UR" in op.mem_addr_offset else False
                #         r_base = "R" in op.reg

                #         if (not ur_offset) or (ur_offset and r_base and "64" in op.suffix) and (not op.is_rz):
                #             new_insts.append(self.create_pack64(op, inst.parent))
                #             count += 1


                #         if ur_offset:
                #             offsetOp = Operand(op.mem_addr_offset, op.mem_addr_offset, None, -1, True, False, False)
                #             inst1 = self.create_pack64(offsetOp, inst.parent)
                #             inst2 = self.create_iadd64(op, offsetOp, inst)
                #             new_insts.append(inst1)
                #             new_insts.append(inst2)
                #             count += 1

                #     new_insts.append(inst)

                # block.instructions = new_insts
                    # IMAD.WIDE.U32 R3, R11, R16, R8, where R8 is a 64 bit value
                    if inst.opcodes[0] == "IMAD" and "WIDE" in inst.opcodes:
                        uses = inst.get_uses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        valOp = uses[-1]
                        if not valOp.is_writable_reg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = valOp.clone()
                        src_op_upper_name = self._next_hi_name(valOp.reg)
                        src_op_upper = Operand.from_reg(
                            src_op_upper_name,
                            src_op_upper_name
                        )

                        new_reg = src_op_lower.name + "_int64_" + str(count)
                        dst_op = Operand.from_reg(new_reg, new_reg)

                        pack_val_inst = Instruction(
                            id=f"{inst.id}_pack64_val",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_val_inst)
                        valOp.set_reg(new_reg)

                    # MATCH.ANY.U64 R35 = R13, where R13 is a 64 bit value
                    if inst.opcodes[0] == "MATCH":

                        if "U64" not in inst.opcodes:
                            new_insts.append(inst)
                            continue

                        uses = inst.get_uses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        valOp = uses[0]
                        if not valOp.is_writable_reg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = valOp.clone()
                        src_op_upper_name = self._next_hi_name(valOp.reg)
                        src_op_upper = Operand.from_reg(
                            src_op_upper_name,
                            src_op_upper_name
                        )

                        new_reg = src_op_lower.name + "_int64_" + str(count)
                        dst_op = Operand.from_reg(new_reg, new_reg)

                        pack_val_inst = Instruction(
                            id=f"{inst.id}_pack64_val",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_val_inst)
                        valOp.set_reg(new_reg)


                    # RED.E.ADD.STRONG.GPU [R2], R5 ;
                    if inst.opcodes[0] == "RED":
                        uses = inst.get_uses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        addrOp = uses[0]
                        if not addrOp.is_writable_reg:
                            new_insts.append(inst)
                            continue

                        # Insert PACK64 instruction
                        src_op_lower = Operand.from_reg(
                            addrOp.reg,
                            addrOp.reg,
                            addrOp.suffix
                        )
                        src_op_upper_name = self._next_hi_name(addrOp.reg)
                        src_op_upper = Operand.from_reg(
                            src_op_upper_name,
                            src_op_upper_name,
                            src_op_lower.suffix
                        )

                        new_reg = src_op_lower.name + "_int64_" + str(count)
                        dst_op = Operand.from_reg(new_reg, new_reg)

                        pack_addr_inst = Instruction(
                            id=f"{inst.id}_pack64_addr",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(pack_addr_inst)
                        addrOp.set_reg(new_reg)

                    if (inst.is_global_load() or inst.is_global_store()):

                        uses = inst.get_uses()
                        if not uses:
                            new_insts.append(inst)
                            continue

                        addrOp = uses[0]

                        src_op_lower = Operand.from_reg(
                            addrOp.reg,
                            addrOp.reg,
                            addrOp.suffix
                        )

                        src_op_upper_name = self._next_hi_name(addrOp.reg)

                        src_op_upper = Operand.from_reg(
                            src_op_upper_name,
                            src_op_upper_name,
                            src_op_lower.suffix
                        )

                        new_reg = src_op_lower.name + "_int64_" + str(count)
                        addrOp.set_reg(new_reg)
                        dst_op = Operand.from_reg(new_reg, new_reg)

                        cast_inst = Instruction(
                            id=f"{inst.id}_pack64",
                            opcodes=["PACK64"],
                            operands=[
                                dst_op,
                                src_op_lower,
                                src_op_upper
                            ],
                            parentBB=inst.parent
                        )
                        count += 1
                        new_insts.append(cast_inst)

                        if addrOp.is_mem_addr and isinstance(addrOp.offset_value, str) and 'UR' in addrOp.offset_value:

                            off = addrOp.offset_value
                            ur_lo = Operand.from_reg(off, off)
                            ur_hi_name = self._next_hi_name(off)
                            ur_hi = Operand.from_reg(ur_hi_name, ur_hi_name)
                            ur64_name = f"{off}_int64"
                            ur64 = Operand.from_reg(ur64_name, ur64_name)

                            new_insts.append(Instruction(
                                id=f"{inst.id}_pack64_{off}",
                                opcodes=["PACK64"],
                                operands=[ur64, ur_lo, ur_hi],
                                parentBB=inst.parent
                            ))
                            count += 1

                            sub_op_name = f"{src_op_lower.name}_usub"
                            sub_op = Operand.from_reg(sub_op_name, sub_op_name)
                            base_op = Operand.from_reg(new_reg, new_reg)
                            offset_op = Operand.from_reg(ur64_name, ur64_name)

                            new_insts.append(Instruction(
                                id=f"{inst.id}_iadd64_usub",
                                opcodes=["IADD64"],
                                operands=[sub_op, base_op, offset_op],
                                parentBB=inst.parent
                            ))

                            # Rewrite pointer to the uniform-substituted 64-bit base and clear offset
                            addrOp.replace(sub_op)

                    # If this is a 64-bit store, pack the value operand STG.64 [R2], R6 => PACK64 R6_int64, R6 R5
                    if inst.is_store() and ('64' in inst.opcodes):
                        uses = inst.get_uses()
                        if len(uses) < 2:
                            new_insts.append(inst)
                            continue

                        valOp = uses[-1]
                        if valOp.is_reg and not valOp.is_rz and self._is_simple_hw_reg(valOp.reg):
                            src_op_lower = valOp.clone()
                            src_op_upper_name = self._next_hi_name(valOp.reg)
                            src_op_upper = Operand.from_reg(
                                src_op_upper_name,
                                src_op_upper_name
                            )

                            new_reg = src_op_lower.name + "_int64_" + str(count)
                            dst_op = Operand.from_reg(new_reg, new_reg)

                            pack_val_inst = Instruction(
                                id=f"{inst.id}_pack64_val",
                                opcodes=["PACK64"],
                                operands=[
                                    dst_op,
                                    src_op_lower,
                                    src_op_upper
                                ],
                                parentBB=inst.parent
                            )
                            count += 1
                            new_insts.append(pack_val_inst)
                            valOp.set_reg(new_reg)
                    new_insts.append(inst)
                block.instructions = new_insts

        self.total_pack64 = count
        print(f"Total PACK64 instructions added: {count}")
        print("=== End of Pack64 Transformation ===")
