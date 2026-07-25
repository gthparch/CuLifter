"""Fold a memory operand's second address register (`index_reg`) into a tracked
add, so the full effective address survives SSA/DCE and resolves at lift time.

ptxas emits two-register memory addresses as `[Rbase + Rindex]` (e.g. the SM90
shared-window mbarrier address `[R2 + UR4]` = SR_SWINLO + in-region offset). The
parser stores the second register in `Operand.index_reg`, which is NOT a tracked
use operand — DefUse/DCE never see it, so DCE removes the chain that computes it
and the lifter (which reads only the base) silently uses `base + 0`. For a
mbarrier that collapses init/arrive/try_wait to different keys and the consumer's
wait never observes the producer's arrive (deadlock).

This pass runs BEFORE SSA: for every operand with an `index_reg`, it inserts
`IADD3 fresh, base, index, RZ` ahead of the instruction and rewrites the operand
to the single register `fresh`. Now `base` and `index` are real use operands of
the IADD3 — SSA versions them, DCE keeps their defs, and `_get_val` on the
rewritten operand returns the full `base + index`. General: applies to any
instruction whose memory operand carries an `index_reg` (not special-cased to
SYNCS). reg-group (64-bit pair) bases and descriptor operands are left alone.
"""
from sir.instruction import Instruction
from sir.operand import Operand
from transform.transform import SaSSTransform


class MemIndexFold(SaSSTransform):
    def apply(self, module):
        super().apply(module)
        self.changed = False
        for func in module.functions:
            for block in func.blocks:
                new_insts = []
                for inst in block.instructions:
                    for oi, op in enumerate(inst.operands):
                        if not getattr(op, "is_mem_addr", False):
                            continue
                        idx = getattr(op, "index_reg", None)
                        if not idx:
                            continue
                        if getattr(op, "reg_group", None):
                            continue  # 64-bit pair base: not a simple [r+r]
                        base = op.reg
                        # Unique per (instruction, operand) so two indexed mem
                        # operands on one instruction don't collide.
                        tag = f"{inst.id}_{oi}"
                        fresh = f"{base}_{idx}_memidx_{tag}"
                        add_inst = Instruction(
                            id=f"memidx_{tag}",
                            opcodes=["IADD3"],
                            operands=[
                                Operand.from_reg(fresh, fresh),
                                Operand.from_reg(base, base),
                                Operand.from_reg(idx, idx),
                                Operand.from_reg("RZ", "RZ"),
                            ],
                            parentBB=block,
                        )
                        new_insts.append(add_inst)
                        op.set_reg(fresh)
                        op.index_reg = None
                        self.changed = True
                    new_insts.append(inst)
                block.instructions = new_insts
