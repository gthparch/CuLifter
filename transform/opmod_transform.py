from collections import Counter

from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class OpModTransform(SaSSTransform):
    ADDRESS_SCALE_SHIFTS = {"X2": 1, "X4": 2, "X8": 3, "X16": 4}

    # 32-bit address spaces (shared/local) whose [Rbase+URidx] operands can be
    # folded with a plain 32-bit add. 64-bit global forms ([R16.64+UR4],
    # already expanded to a reg-pair group at parse time) are folded below
    # with an IADD64 over the base pair and the UR pair.
    IDX_MEM_OPS = {"LDS", "STS", "LDSM", "STSM", "ATOMS", "LDL", "STL", "LDGSTS", "SYNCS"}

    @staticmethod
    def _next_reg_name(reg):
        """R16 -> R17, UR4 -> UR5; RZ/URZ map to themselves (pair of zeros)."""
        if reg in ("RZ", "URZ"):
            return reg
        prefix_len = 2 if reg.startswith("UR") else 1
        return f"{reg[:prefix_len]}{int(reg[prefix_len:]) + 1}"

    def apply(self, module):
        print("=== Start of OpModeTransform ===")

        scale_counts = Counter()
        total_idx_folds = 0
        new_regs_count = {}

        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    inserted_here = 0
                    # Scan all operands; 
                    for op in inst.operands:
                        # Scale the base before folding the index. The immediate
                        # displacement remains attached to the memory operand.
                        if (op.is_mem_addr and op.is_reg and not op.is_reg_group
                                and op.suffix in self.ADDRESS_SCALE_SHIFTS):
                            scale = op.suffix.lower()
                            shift = self.ADDRESS_SCALE_SHIFTS[op.suffix]
                            base_reg = op.reg  # underlying register name (e.g., R7)

                            # Source operand (as a plain register, not a memory address)
                            src_op = Operand.from_reg(base_reg, base_reg)

                            # Unique destination register name
                            base_dst = f"{base_reg}_{scale}"
                            new_regs_count[base_dst] = new_regs_count.get(base_dst, 0) + 1
                            dst_name = f"{base_dst}_{new_regs_count[base_dst]}"
                            dst_op = Operand.from_reg(dst_name, dst_name)

                            imm_op = Operand.from_immediate(hex(shift), shift)

                            shl_inst = Instruction(
                                id=f"{inst.id}_{scale}",
                                opcodes=["SHL"],
                                operands=[dst_op, src_op, imm_op],
                                parentBB=inst.parent
                            )

                            # Insert before the current instruction
                            new_insts.append(shl_inst)
                            inserted_here += 1
                            scale_counts[op.suffix] += 1

                            # Rewrite memory operand to use the new register (and drop suffix)
                            op.set_reg(dst_name)
                            op.suffix = None

                        # Materialize 64-bit reg+reg addressing
                        # ([R16.64+UR4] / [RZ.U32+UR6], parse-expanded to a
                        # base reg-pair group) into an explicit IADD64 of the
                        # base pair and the UR pair. Nothing downstream reads
                        # index_reg, so without this the +UR term is silently
                        # dropped (CB_hgemm epilogue: round-4 STG.E.128
                        # [R16.64+UR4] landed on round-3's columns and its
                        # beta C-load re-read freshly written D).
                        if (op.is_mem_addr and op.index_reg
                                and op.is_reg_group and len(op.reg_group) == 2):
                            lo = op.reg_group[0].reg
                            hi = op.reg_group[1].reg
                            idx_lo = op.index_reg
                            idx_hi = self._next_reg_name(idx_lo)

                            base_grp = Operand.from_reg_group(
                                f"{hi}:{lo}", [lo, hi])
                            idx_grp = Operand.from_reg_group(
                                f"{idx_hi}:{idx_lo}", [idx_lo, idx_hi])

                            base_dst = f"{lo}_idx64"
                            new_regs_count[base_dst] = new_regs_count.get(base_dst, 0) + 1
                            n = new_regs_count[base_dst]
                            dst_lo = f"{base_dst}_{n}"
                            dst_hi = f"{base_dst}h_{n}"
                            dst_grp = Operand.from_reg_group(
                                f"{dst_hi}:{dst_lo}", [dst_lo, dst_hi])

                            add64 = Instruction(
                                id=f"{inst.id}_idx64",
                                opcodes=["IADD64"],
                                operands=[dst_grp, base_grp, idx_grp],
                                parentBB=inst.parent
                            )
                            new_insts.append(add64)
                            inserted_here += 1
                            total_idx_folds += 1

                            op.reg_group[0].set_reg(dst_lo)
                            op.reg_group[1].set_reg(dst_hi)
                            op.index_reg = None
                            continue

                        # Materialize reg+reg addressing ([R13+UR4]) into an
                        # explicit IADD3: nothing downstream (SSA, Pack64,
                        # lifters) reads index_reg, so it would be silently
                        # dropped — pre-SSA expansion gives the index reg a
                        # real use that gets versioned and kept alive.
                        if (op.is_mem_addr and op.index_reg
                                and inst.opcodes[0] in self.IDX_MEM_OPS):
                            base_reg = op.reg
                            idx_reg = op.index_reg

                            base_op = Operand.from_reg(base_reg, base_reg)
                            idx_op = Operand.from_reg(idx_reg, idx_reg)
                            rz_op = Operand.from_reg("RZ", "RZ")

                            base_dst = f"{base_reg}_idx"
                            new_regs_count[base_dst] = new_regs_count.get(base_dst, 0) + 1
                            dst_name = f"{base_dst}_{new_regs_count[base_dst]}"
                            dst_op = Operand.from_reg(dst_name, dst_name)

                            add_inst = Instruction(
                                id=f"{inst.id}_idx",
                                opcodes=["IADD3"],
                                operands=[dst_op, base_op, idx_op, rz_op],
                                parentBB=inst.parent
                            )

                            new_insts.append(add_inst)
                            inserted_here += 1
                            total_idx_folds += 1

                            op.index_reg = None
                            op.set_reg(dst_name)

                    new_insts.append(inst)

                block.instructions = new_insts

        self.total_x4_expansions = scale_counts["X4"]
        print(f"Total scaled address expansions inserted: {sum(scale_counts.values())}")
        if total_idx_folds:
            print(f"Total index-reg folds inserted: {total_idx_folds}")
        print("=== End of OpModeTransform ===")
