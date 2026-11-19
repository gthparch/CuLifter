from transform.transform import SaSSTransform


class CFGSimplify(SaSSTransform):
    _IGNORABLE_OPCODES = {"BREAK", "SSY", "SYNC", "WARPSYNC", "BSSY", "BSYNC", "BMOV"}

    def apply(self, module):
        for func in module.functions:
            self._simplify_function(func)

    def _simplify_function(self, func):
        if not func.blocks or len(func.blocks) < 2:
            return

        # Phase 1: Remove unreachable blocks (no predecessors, except entry).
        self._remove_unreachable(func)

        # Phase 2: Collapse trampoline blocks (single unconditional BRA).
        changed = True
        while changed:
            changed = False
            for bb in list(func.blocks)[1:]:
                if not self._is_uncond_bra_trampoline(bb):
                    continue

                succ = bb.succs[0]
                if succ is bb:
                    continue

                preds = list(bb.preds)
                if any(pred in succ.preds and pred is not bb for pred in preds):
                    continue

                succ_phi_insts = []
                for inst in succ.instructions:
                    if inst.is_phi():
                        succ_phi_insts.append(inst)
                    else:
                        break

                old_pred_idx = succ.preds.index(bb) if bb in succ.preds else None
                if old_pred_idx is not None:
                    for phi in succ_phi_insts:
                        incoming = phi.operands[old_pred_idx + 1]
                        phi.operands = (
                            phi.operands[: old_pred_idx + 1]
                            + [incoming.clone() for _ in preds]
                            + phi.operands[old_pred_idx + 2 :]
                        )
                    succ.preds = (
                        succ.preds[:old_pred_idx] + preds + succ.preds[old_pred_idx + 1 :]
                    )

                for pred in preds:
                    self._redirect_edge(pred, bb, succ)

                if bb in func.blocks:
                    func.blocks.remove(bb)
                changed = True

        # Phase 3: Reorder blocks into reverse-postorder (RPO).
        # This ensures that for acyclic edges the definition block
        # appears before all use blocks, which is required by the
        # LLVM IR emitter (it processes blocks in list order).
        self._reorder_rpo(func)

    def _reorder_rpo(self, func):
        """Sort func.blocks into reverse-postorder of the CFG."""
        if not func.blocks:
            return
        entry = func.blocks[0]
        visited = set()
        order = []

        stack = [(entry, iter(entry.succs))]
        visited.add(id(entry))

        while stack:
            parent, children = stack[-1]
            try:
                child = next(children)
                if id(child) not in visited:
                    visited.add(id(child))
                    stack.append((child, iter(child.succs)))
            except StopIteration:
                stack.pop()
                order.append(parent)

        rpo = list(reversed(order))

        # Append any blocks not reachable from entry (shouldn't exist
        # after _remove_unreachable, but keep for safety).
        rpo_set = set(id(b) for b in rpo)
        for bb in func.blocks:
            if id(bb) not in rpo_set:
                rpo.append(bb)

        func.blocks = rpo

    def _remove_unreachable(self, func):
        """Remove blocks not reachable from the entry block.

        Forward BFS from the entry block marks reachable blocks; everything
        else is removed along with its incident edges. This handles both
        (a) blocks with no predecessors and (b) strongly-connected components
        where every block has a predecessor but the SCC has no path from
        entry. The pure empty-preds algorithm used to miss case (b) — see
        BUG-070: an SCC whose static entry edge was orphaned by the CFG
        builder persisted through simplify, broke SSA's dominator-tree
        walk (which only renames entry-reachable blocks), and then tripped
        DefUseAnalysis's duplicate-def check on the un-renamed defs.
        """
        if not func.blocks:
            return
        entry = func.blocks[0]
        reachable = {id(entry)}
        worklist = [entry]
        while worklist:
            bb = worklist.pop()
            for succ in bb.succs:
                if id(succ) not in reachable:
                    reachable.add(id(succ))
                    worklist.append(succ)

        for bb in list(func.blocks):
            if id(bb) in reachable:
                continue
            for succ in list(bb.succs):
                if bb in succ.preds:
                    succ.preds.remove(bb)
            for pred in list(bb.preds):
                if bb in pred.succs:
                    pred.succs.remove(bb)
            bb.succs = []
            bb.preds = []
            func.blocks.remove(bb)

    def _is_uncond_bra_trampoline(self, bb):
        if len(bb.succs) != 1:
            return False

        if not bb.instructions:
            return False

        effective_insts = [
            inst
            for inst in bb.instructions
            if inst.opcodes and inst.opcodes[0] not in self._IGNORABLE_OPCODES
        ]
        if len(effective_insts) != 1:
            return False

        inst = effective_insts[0]
        if not inst.opcodes or inst.opcodes[0] != "BRA":
            return False
        if inst.predicated() or inst.is_indirect_branch():
            return False
        # BRA with a predicate register operand (e.g. BRA P1, target from
        # @!P0 BRA P1, target after pflag stripping) is conditional, not a trampoline.
        if inst.operands and len(inst.operands) >= 2 and inst.operands[0].is_predicate_reg:
            return False
        if not inst.operands:
            return False
        target_op = inst.operands[-1]
        if not getattr(target_op, "is_immediate", False):
            return False
        if target_op.immediate_value is None:
            return False
        if target_op.immediate_value != bb.succs[0].addr:
            return False

        for extra in bb.instructions:
            if extra is inst:
                continue
            if extra.predicated():
                return False

        return True

    def _redirect_edge(self, pred, old_succ, new_succ):
        if pred is None or old_succ is None or new_succ is None or pred is old_succ:
            return

        if pred in old_succ.preds:
            old_succ.preds.remove(pred)
        if old_succ in pred.succs:
            pred.succs = [new_succ if succ is old_succ else succ for succ in pred.succs]

        term = pred.get_terminator()
        if term is None or not term.opcodes:
            return

        # A direct CALL can have both a target edge and a fallthrough edge.
        # Keep the literal target in sync when its trampoline is collapsed.
        if term.operands and (
            term.opcodes[0] == "BRA" or (term.is_call() and len(term.operands) == 1)
        ):
            target_op = term.operands[-1]
            if getattr(target_op, "is_immediate", False) and target_op.immediate_value == old_succ.addr:
                target_op.name = new_succ.addr_content
                target_op.immediate_value = new_succ.addr

        if term.opcodes and term.opcodes[0] == "PBRA":
            for op in term.operands[1:3]:
                if getattr(op, "is_immediate", False) and op.immediate_value == old_succ.addr:
                    op.name = new_succ.addr_content
                    op.immediate_value = new_succ.addr

        if term.opcodes[0] == "SWITCH":
            pred.succs = self._dedup_by_addr(pred.succs)
            # Keep the ordered switch_targets list in sync with the redirect.
            if hasattr(term, "switch_targets"):
                term.switch_targets = [
                    new_succ if tgt is old_succ else tgt
                    for tgt in term.switch_targets
                ]

    def _dedup_by_addr(self, blocks):
        seen = set()
        result = []
        for bb in blocks:
            if bb is None:
                continue
            addr = getattr(bb, "addr", None)
            if addr in seen:
                continue
            seen.add(addr)
            result.append(bb)
        return result
