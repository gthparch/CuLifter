from transform.transform import SaSSTransform
from sir.instruction import Instruction
from sir.operand import Operand
from collections import deque, defaultdict

from transform.domtree import DominatorTree

class SSA(SaSSTransform):

    def apply(self, module):
        print("=== Start of SSA ===")

        for func in module.functions:
            if not func.blocks:
                continue

            dom_tree = self.compute_dominator_tree(func)
            self.insert_phi_nodes(func, dom_tree)
            self.rename_variables(func, dom_tree)
            self.insert_set_zero_for_undefs(func)

        print("=== End of SSA ===")

    def compute_dominator_tree(self, function):
        if not function.blocks:
            return None

        entry_block = function.blocks[0]
        return DominatorTree(entry_block, function.blocks)

    def insert_phi_nodes(self, function, dom_tree):
        # Iterated dominance frontier algorithm on register defs.
        def_sites = defaultdict(set)
        all_vars = set()
        for block in function.blocks:
            for inst in block.instructions:
                # Only regs that are writable should be tracked
                for def_op in inst.get_defs():
                    if def_op and def_op.is_writable_reg:
                        if def_op.is_reg_group:
                            for sub_op in def_op.reg_group:
                                var_name = self.get_original_reg(sub_op.reg)
                                if var_name and not self.should_skip_variable(var_name, None, None):
                                    def_sites[var_name].add(block)
                                    all_vars.add(var_name)
                        else:
                            var_name = self.get_original_reg(def_op.reg)
                            if var_name and not self.should_skip_variable(var_name, None, None):
                                def_sites[var_name].add(block)
                                all_vars.add(var_name)

        # DETERMINISM: all_vars / def_sites / dominance_frontier are sets, and
        # set-of-str iteration order is randomized per process (PYTHONHASHSEED)
        # — PHI insertion order was a per-run coin flip (the long-standing
        # "PHI nondeterminism" quirk), and some orderings wire downstream
        # consumers to non-dominating defs (so.69 bn_bw: the whole-module
        # lift failed IR verification while a standalone lift of the same
        # function compiled clean). Iterate vars sorted, blocks in CFG order.
        block_order = {b: i for i, b in enumerate(function.blocks)}
        for var in sorted(all_vars):
            worklist = sorted(def_sites[var], key=lambda b: block_order.get(b, 1 << 30))
            phi_added_at = set()

            while worklist:
                block = worklist.pop(0)
                for df_block in sorted(dom_tree.dominance_frontier.get(block, []),
                                       key=lambda b: block_order.get(b, 1 << 30)):
                    # if self.predicatesContradict(block, df_block):
                    #     continue

                    if df_block not in phi_added_at:
                        phi = self.create_phi_node(var, df_block)
                        # Insert PHI at the top of the block
                        df_block.instructions.insert(0, phi)
                        phi_added_at.add(df_block)
                        # Treat the PHI as a new def site of var
                        worklist.append(df_block)

    def should_skip_variable(self, var, reaching_defs, block):
        # Skip only special non-SSA globals: PT/UPT and zero registers RZ/URZ.
        # Do NOT skip predicate registers like P0/UP1, they should be in SSA.
        skip_set = {"PT", "UPT", "RZ", "URZ"}
        return var in skip_set

    def create_phi_node(self, reg, block):
        dest_op = Operand.from_reg(reg, reg)

        operands = [dest_op]
        # Allocate one incoming operand per predecessor (will be filled in rename)
        for _ in block.preds:
            src_op = Operand.from_reg(reg, reg)
            operands.append(src_op)

        phi = Instruction(
            id=f"phi_{reg}_{block.addr_content}",
            opcodes=["PHI"],
            operands=operands,
            parentBB=block
        )
        return phi

    def rename_variables(self, function, dom_tree):
        self.var_stacks = defaultdict(list)
        self.ssa_counter = defaultdict(int)
        self.undefined_ssa = {}
        self.entry_block = dom_tree.entry if dom_tree else None

        self.rename_block(dom_tree.entry, dom_tree)

    def rename_block(self, block, dom_tree):
        if block is None:
            return
        stack = [(block, 0, None)]

        while stack:
            cur_block, phase, pushed = stack.pop()

            if phase == 0:
                pushed = defaultdict(list)

                self.process_phi_nodes(cur_block, pushed)
                self.process_instructions(cur_block, pushed)

                self.update_successor_phis(cur_block)

                stack.append((cur_block, 1, pushed))

                children = dom_tree.get_children(cur_block) if dom_tree else []
                for child in reversed(children):
                    stack.append((child, 0, None))
            else:
                self.restore_stacks(pushed)

    def process_phi_nodes(self, block, pushed):
        for inst in block.instructions:
            if not inst.is_phi():
                continue

            dest_op = inst.operands[0]
            if dest_op.is_writable_reg:
                if dest_op.is_reg_group:
                    for sub_op in dest_op.reg_group:
                        original_reg = self.get_original_reg(sub_op.reg)
                        new_name = self.generate_ssa_name(original_reg)
                        sub_op.set_reg(new_name)
                        sub_op.ir_reg_name = None
                        self.var_stacks[original_reg].append(new_name)
                        pushed[original_reg].append(new_name)
                else:
                    original_reg = self.get_original_reg(dest_op.reg)
                    new_name = self.generate_ssa_name(original_reg)

                    dest_op.set_reg(new_name)
                    dest_op.ir_reg_name = None

                    self.var_stacks[original_reg].append(new_name)
                    pushed[original_reg].append(new_name)

    def process_instructions(self, block, pushed):
        for inst in block.instructions:
            if inst.is_phi():
                continue

            # Rename uses (including predicate flags)
            for use_op in inst.get_uses_with_predicate():
                if not use_op.is_writable_reg:
                    continue

                if use_op.is_reg_group:
                    for sub_op in use_op.reg_group:
                        original_reg = self.get_original_reg(sub_op.reg)
                        current_name = self.get_current_name_or_version0(original_reg)
                        sub_op.set_reg(current_name)
                        sub_op.ir_reg_name = None
                else:
                    original_reg = self.get_original_reg(use_op.reg)
                    current_name = self.get_current_name_or_version0(original_reg)
                    use_op.set_reg(current_name)
                    use_op.ir_reg_name = None

            # Rename defs
            is_predicated_def = (
                inst.predicated()
                and not inst.is_branch()
                and not inst.is_exit()
                and inst.opcodes[0] not in (
                    "BRA", "EXIT", "RET", "PBRA", "BREAK",
                    "BSSY", "BSYNC", "SSY", "SYNC", "NOP",
                )
            )

            sel_inserts = []  # (insert_after_inst, sel_instruction)

            for def_op in inst.get_defs():
                if def_op and def_op.is_writable_reg:
                    if def_op.is_reg_group:
                        for sub_op in def_op.reg_group:
                            original_reg = self.get_original_reg(sub_op.reg)
                            if is_predicated_def and not self.should_skip_variable(original_reg, None, None):
                                self._rename_predicated_def(
                                    sub_op, original_reg, inst, block, pushed, sel_inserts
                                )
                            else:
                                new_name = self.generate_ssa_name(original_reg)
                                sub_op.set_reg(new_name)
                                sub_op.ir_reg_name = None
                                self.var_stacks[original_reg].append(new_name)
                                pushed[original_reg].append(new_name)
                    else:
                        original_reg = self.get_original_reg(def_op.reg)
                        if is_predicated_def and not self.should_skip_variable(original_reg, None, None):
                            self._rename_predicated_def(
                                def_op, original_reg, inst, block, pushed, sel_inserts
                            )
                        else:
                            new_name = self.generate_ssa_name(original_reg)
                            def_op.set_reg(new_name)
                            def_op.ir_reg_name = None
                            self.var_stacks[original_reg].append(new_name)
                            pushed[original_reg].append(new_name)

            # Insert SEL instructions after the predicated instruction
            if sel_inserts:
                idx = block.instructions.index(inst)
                for offset, sel in enumerate(sel_inserts, 1):
                    block.instructions.insert(idx + offset, sel)

    def _rename_predicated_def(self, def_op, original_reg, inst, block, pushed, sel_inserts):
        """Handle a predicated def by inserting a SEL to merge old and new values.

        For `@P0 LDC R8, src`:
          1. old_name = current SSA name of R8 (value before this instruction)
          2. new_name = fresh SSA version for R8 (the predicated write)
          3. merged_name = fresh SSA version (SEL result)
          4. Insert: SEL merged_name = new_name, old_name, P0
          5. Push merged_name on the stack (not new_name)
        """
        old_name = self.get_current_name_or_version0(original_reg)
        new_name = self.generate_ssa_name(original_reg)

        def_op.set_reg(new_name)
        def_op.ir_reg_name = None

        # Create SEL: merged = P ? new : old  (or P ? old : new if negated)
        merged_name = self.generate_ssa_name(original_reg)
        pflag = inst.pflag
        is_neg = pflag.is_not_reg if pflag else False

        dest = Operand.from_reg(merged_name, merged_name)
        if is_neg:
            # @!P: write happens when P is false → merged = P ? old : new
            src_true = Operand.from_reg(old_name, old_name)
            src_false = Operand.from_reg(new_name, new_name)
        else:
            # @P: write happens when P is true → merged = P ? new : old
            src_true = Operand.from_reg(new_name, new_name)
            src_false = Operand.from_reg(old_name, old_name)

        # The predicate operand — use the already-renamed version from the pflag
        pred_op = Operand.from_reg(pflag.reg, pflag.reg, prefix=None)

        sel = Instruction(
            id=f"{inst.id}_predsel_{original_reg}",
            opcodes=["SEL"],
            operands=[dest, src_true, src_false, pred_op],
            parentBB=block,
        )
        sel_inserts.append(sel)

        # Push the MERGED version, not the raw new version
        self.var_stacks[original_reg].append(merged_name)
        pushed[original_reg].append(merged_name)

    def update_successor_phis(self, block):
        for succ in block.succs:
            # Find our position in succ's predecessor list
            pred_idx = succ.preds.index(block)

            for inst in succ.instructions:
                if not inst.is_phi():
                    continue

                dest_op = inst.operands[0]
                if pred_idx + 1 < len(inst.operands):
                    phi_op = inst.operands[pred_idx + 1]
                    if dest_op.is_reg_group and phi_op.is_reg_group:
                        for i, sub_dest in enumerate(dest_op.reg_group):
                            if i < len(phi_op.reg_group):
                                original_reg = self.get_original_reg(sub_dest.reg)
                                current_name = self.get_current_name_or_version0(original_reg)
                                phi_op.reg_group[i].set_reg(current_name)
                                phi_op.reg_group[i].ir_reg_name = None
                    elif not dest_op.is_reg_group and not phi_op.is_reg_group:
                        original_reg = self.get_original_reg(dest_op.reg)
                        current_name = self.get_current_name_or_version0(original_reg)
                        phi_op.set_reg(current_name)
                        phi_op.ir_reg_name = None

    def restore_stacks(self, pushed):
        for var, names in pushed.items():
            for _ in names:
                if self.var_stacks[var]:
                    self.var_stacks[var].pop()

    def get_original_reg(self, reg_name):
        if not reg_name:
            return None
        if '@' in reg_name:
            return reg_name.split('@')[0]
        return reg_name

    def generate_ssa_name(self, original_reg):
        if not original_reg:
            return None
        self.ssa_counter[original_reg] += 1
        return f"{original_reg}@{self.ssa_counter[original_reg]}"

    def get_current_name_or_version0(self, original_reg):
        if not original_reg:
            return None

        stack = self.var_stacks[original_reg]
        if stack:
            return stack[-1]

        zero_name = self.undefined_ssa.get(original_reg)
        if not zero_name:
            zero_name = f"{original_reg}@0"
            self.undefined_ssa[original_reg] = zero_name

        stack.append(zero_name)
        return zero_name

    def insert_set_zero_for_undefs(self, function):
        

        entry_block = function.blocks[0]

        # Insert after phi
        insert_idx = 0
        while insert_idx < len(entry_block.instructions) and entry_block.instructions[insert_idx].is_phi():
            insert_idx += 1

        new_insts = []
        for original_reg in sorted(self.undefined_ssa):
            ssa_name = self.undefined_ssa[original_reg]
            dest_op = Operand.from_reg(original_reg, original_reg)
            dest_op.set_reg(ssa_name)
            dest_op.ir_reg_name = None

            inst_id = f"setzero_{ssa_name}"
            if entry_block.addr_content:
                inst_id = f"{entry_block.addr_content}_{inst_id}"

            new_insts.append(
                Instruction(
                    id=inst_id,
                    opcodes=["SETZERO"],
                    operands=[dest_op],
                    parentBB=entry_block
                )
            )

        entry_block.instructions[insert_idx:insert_idx] = new_insts
        self.undefined_ssa = {}
