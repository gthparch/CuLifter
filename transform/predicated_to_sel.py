"""Pre-SSA transform: convert predicated defs to unconditional def + SEL.

For each predicated instruction like `@P0 LDC R8, src`:
  1. Strip the predicate from the instruction: `LDC R8_tmp, src`
  2. Insert a SEL after: `SEL R8, R8_tmp, R8_old, P0`

This ensures SSA correctly models that R8 may or may not be updated,
preventing the bug where a predicated def unconditionally pushes a
new SSA version (BUG-020).

Only applies to non-branch, non-exit predicated instructions that define registers.
"""

from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand


class PredicatedToSel(SaSSTransform):
    def apply(self, module):
        count = 0
        for func in module.functions:
            count += self._process_function(func)
        if count:
            print(f"PredicatedToSel: inserted {count} SEL instructions")

    def _process_function(self, func):
        count = 0
        new_instructions = []

        for inst in func.instructions:
            if not inst.pflag:
                new_instructions.append(inst)
                continue

            # Skip branches, exits, and instructions without register defs
            if inst.is_branch() or inst.is_exit() or inst.opcodes[0] in (
                "BRA", "EXIT", "RET", "PBRA", "BREAK", "BSSY", "BSYNC",
                "SSY", "SYNC", "WARPSYNC", "NOP", "BMOV",
            ):
                new_instructions.append(inst)
                continue

            defs = [d for d in inst.get_defs() if d and d.is_writable_reg and not d.is_pt]
            if not defs:
                new_instructions.append(inst)
                continue

            # This instruction has predicated register defs.
            # For each def register, insert a SEL after the instruction.
            pflag = inst.pflag
            is_neg = pflag.is_not_reg  # @!P0 means execute if P0 is false

            # Strip predicate from the original instruction
            inst.pflag = None

            new_instructions.append(inst)

            # Insert SEL for each def
            for def_op in defs:
                if def_op.is_reg_group:
                    # For register groups, insert SEL for each sub-register
                    for sub_op in def_op.reg_group:
                        sel = self._make_sel(sub_op.reg, pflag, is_neg, inst.id)
                        new_instructions.append(sel)
                        count += 1
                else:
                    sel = self._make_sel(def_op.reg, pflag, is_neg, inst.id)
                    new_instructions.append(sel)
                    count += 1

        func.instructions = new_instructions
        return count

    def _make_sel(self, reg, pflag, is_neg, parent_id):
        """Create: SEL Rdest, Rnew, Rold, P
        If @P0 (not negated): SEL R, R_new, R_old, P0 → if P0: R_new, else R_old
        If @!P0 (negated):    SEL R, R_old, R_new, P0 → if P0: R_old, else R_new
        """
        dest = Operand.from_reg(reg, reg)
        # new_val = current value (just written by the predicated instruction)
        new_val = Operand.from_reg(reg, reg)
        # old_val = previous value (what R held before the predicated write)
        old_val = Operand.from_reg(reg, reg)
        # predicate
        pred = pflag.clone()

        if is_neg:
            # @!P: execute if P is false → keep old when P is true
            operands = [dest, old_val, new_val, pred]
        else:
            # @P: execute if P is true → keep new when P is true
            operands = [dest, new_val, old_val, pred]

        sel = Instruction(
            id=f"{parent_id}_predsel_{reg}",
            opcodes=["SEL"],
            operands=operands,
        )
        return sel
