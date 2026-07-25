from transform.transform import SaSSTransform
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.operand import Operand


class BuildCFG(SaSSTransform):
    def apply(self, module):
        for func in module.functions:
            func.blocks = self.create_cfg(func.instructions, func)

    def create_cfg(self, insts, curr_func):
        addr_set = {int(inst.id, 16) for inst in insts}
        addr_list = sorted(a for a in addr_set)

        def _align_up_to_inst(addr_hex):
            addr = int(addr_hex, 16)
            max_addr = addr_list[-1]

            while addr not in addr_set and addr <= max_addr:
                addr += 0x8
            return addr

        for inst in insts:
            if inst.is_branch() or inst.is_call():
                target_op = inst.operands[-1]
                if target_op.is_reg:
                    continue
                # nvdisasm single-file CALLs can target a named helper symbol
                # (e.g. `$__internal_..._div_rn_noftz_f32_slowpath`) rather than
                # a hex address — int(.,16) would crash BuildCFG (division and
                # any kernel with the f32-divide slowpath). Leave symbolic
                # targets as-is for HandleCall/the lifter to resolve.
                target_str = target_op.__str__()
                try:
                    int(target_str, 16)
                except (ValueError, TypeError):
                    continue
                target_addr = _align_up_to_inst(target_str)
                inst.operands[-1].name = format(target_addr, "04x")
                inst.operands[-1].immediate_value = target_addr

        leaders = set()
        predicated_leaders = set()
        curr_pred = None
        for i, inst in enumerate(insts):
            if inst.is_return() or inst.is_exit() or inst.is_call():
                if i + 1 < len(insts):
                    leaders.add(insts[i + 1].id)
            if inst.is_branch():
                if not inst.is_indirect_branch():
                    if inst.operands and not inst.operands[-1].is_reg:
                        leaders.add(inst.operands[-1].name.zfill(4))
                if i + 1 < len(insts):
                    leaders.add(insts[i + 1].id)
            if inst.is_call() and not inst.operands[0].is_reg:
                leaders.add(inst.operands[0].name.zfill(4))
            if str(curr_pred) != str(inst.pflag):
                leaders.add(insts[i].id)
                if inst.predicated():
                    predicated_leaders.add(insts[i].id)
                curr_pred = inst.pflag

        blocks = []
        block_insts = []
        pflag = None
        prev_block = None
        next_block = {}
        block_by_addr = {}
        # LIST, not set: this is iterated (PBRA-split rewiring below), and a
        # set of BasicBlock objects iterates in id()/heap-address order —
        # run-stable but allocation-history-sensitive, so the SAME function
        # produced differently-ordered pred lists depending on which other
        # functions were parsed first (BUG-092 track 2: fn37's preds permuted
        # whenever fn44 was in the module; the PHI operand order downstream
        # follows pred order positionally, eventually flipping SSA
        # reaching-version shapes). Creation order = instruction order =
        # context-independent. Each block is created and added exactly once.
        predicated_blocks = []
        for inst in insts:
            if inst.id in leaders:
                if block_insts:
                    block_id = f"{int(block_insts[0].id, 16):04X}"

                    predicated_block = False
                    if block_insts[0].id in predicated_leaders:
                        predicated_block = True
                        pflag = block_insts[0].pflag.clone()
                        pbra_inst = Instruction(
                            id=f"{block_id}_syn",
                            opcodes=["PBRA"],
                            operands=[block_insts[0].pflag.clone()],
                            parentBB=None,
                            pflag=None,
                        )
                        block_insts.insert(0, pbra_inst)
                        for pred_inst in block_insts:
                            pred_inst.pflag = None

                    block = BasicBlock(block_id, pflag, block_insts)
                    pflag = None
                    blocks.append(block)

                    if prev_block:
                        next_block[prev_block] = block
                    prev_block = block

                    block_by_addr[block.addr_content] = block

                    if predicated_block:
                        predicated_blocks.append(block)

                block_insts = [inst]
            else:
                block_insts.append(inst)

        # Flush remaining instructions as the final block
        if block_insts:
            block_id = f"{int(block_insts[0].id, 16):04X}"

            predicated_block = False
            if block_insts[0].id in predicated_leaders:
                predicated_block = True
                pflag = block_insts[0].pflag.clone()
                pbra_inst = Instruction(
                    id=f"{block_id}_syn",
                    opcodes=["PBRA"],
                    operands=[block_insts[0].pflag.clone()],
                    parentBB=None,
                    pflag=None,
                )
                block_insts.insert(0, pbra_inst)
                for pred_inst in block_insts:
                    pred_inst.pflag = None

            block = BasicBlock(block_id, pflag, block_insts)
            pflag = None
            blocks.append(block)

            if prev_block:
                next_block[prev_block] = block
            prev_block = block

            block_by_addr[block.addr_content] = block

            if predicated_block:
                predicated_blocks.append(block)

        for block in blocks:
            terminator = block.get_terminator()

            if not terminator:
                if block in next_block:
                    succ_block = next_block[block]
                    dest_op = Operand.from_immediate(
                        succ_block.addr_content, int(succ_block.addr_content, 16)
                    )
                    new_inst = Instruction(
                        id=f"{block.addr_content}_syn",
                        opcodes=["BRA"],
                        operands=[dest_op],
                    )
                    block.instructions.append(new_inst)

                    block.add_succ(succ_block)
                    succ_block.add_pred(block)
            elif terminator.is_branch():
                if not terminator.is_indirect_branch():
                    # BRA.DIV branches *only when the warp is divergent*; it is
                    # the compiler's escape to a slow per-lane handler (often a
                    # CALL-based __shfl emulation) around an inline warp-collective
                    # fast path. A semantics-preserving lift assumes full convergent
                    # warps, so the divergence path is never taken — BRA.DIV must
                    # FALL THROUGH to the inline path, not jump to the handler.
                    # (BRA.CONV, by contrast, *is* taken when convergent, so it
                    # stays an unconditional jump to its target.) Treating BRA.DIV
                    # as unconditional-to-target forced the slow path and corrupted
                    # warp-collective results — e.g. nvtiff LZW's cross-warp prefix
                    # scan (each cross-warp SHFL.UP sits behind a BRA.DIV).
                    is_bra_div = (
                        terminator.opcodes
                        and terminator.opcodes[0] == "BRA"
                        and "DIV" in terminator.opcodes
                    )
                    if is_bra_div and block in next_block:
                        fallthrough_block = next_block[block]
                        terminator.operands[-1] = Operand.from_immediate(
                            fallthrough_block.addr_content,
                            int(fallthrough_block.addr_content, 16),
                        )
                        if fallthrough_block not in block.succs:
                            block.add_succ(fallthrough_block)
                            fallthrough_block.add_pred(block)
                        continue

                    dest_op = terminator.operands[-1]
                    succ_block = block_by_addr[dest_op.name]
                    block.add_succ(succ_block)
                    succ_block.add_pred(block)
                    # A BRA needs a fallthrough edge only if it is conditional.
                    # Conditional cases: pflag set, or a predicate register
                    # operand (e.g. BRA P1, target from @!P0 BRA P1 after
                    # pflag stripping in predicated blocks).
                    # BRA.CONV is unconditional — it always jumps to the target.
                    # The reconvergence path back to the fallthrough address is
                    # handled by BSSY/BSYNC (SM90) or SSY/SYNC (SM75).
                    has_pred_operand = (
                        len(terminator.operands) >= 2
                        and terminator.operands[0].is_predicate_reg
                    )
                    needs_fallthrough = (
                        terminator.pflag is not None
                        or has_pred_operand
                    )
                    if needs_fallthrough and block in next_block:
                        fallthrough_block = next_block[block]
                        if fallthrough_block not in block.succs:
                            block.add_succ(fallthrough_block)
                            fallthrough_block.add_pred(block)
            elif terminator.is_return() or terminator.is_exit():
                # A guard-predicate OPERAND makes an EXIT/RET conditional:
                # `@pred EXIT Pg` exits iff (pred && Pg) — the @pred is the
                # PBRA split, Pg gates THIS exit, falling through otherwise.
                # Without a fall-through edge the Pg guard is lost: sheath's
                # `@!P0 EXIT P1` (exit iff x<XMAX && x>=0) drops the x>=0
                # guard, so x<0 particles skip the `alive=false` store, stay
                # live, and crash the next kernel with an OOB scatter.
                # (lift_exit emits the conditional branch — see there.)
                has_pred_operand = (
                    terminator.operands
                    and terminator.operands[0].is_predicate_reg
                    and not terminator.operands[0].is_pt
                )
                if has_pred_operand and block in next_block:
                    fallthrough_block = next_block[block]
                    if fallthrough_block not in block.succs:
                        block.add_succ(fallthrough_block)
                        fallthrough_block.add_pred(block)
            elif terminator.is_call():
                if block in next_block:
                    succ_block = next_block[block]
                    block.add_succ(succ_block)
                    succ_block.add_pred(block)
                # BUG-091: for *predicated* CALL with literal target (e.g.
                # `@P0 CALL.REL.NOINC <addr>`), also add the call target
                # as a CFG successor. The lifter emits a branch to the
                # target (lift_call); without the CFG edge, the SSA pass
                # leaves the target BB with no predecessor and live
                # registers come out as zero at the target. Only do this
                # for predicated CALLs — unpredicated CALLs are handled
                # by HandleCall's device-function inlining; adding the
                # CFG edge for those would conflate the call body with
                # main blocks.
                # Detect "originally predicated CALL" by checking whether
                # the block starts with a synthetic PBRA (which the CFG
                # builder inserts for predicated blocks earlier — see
                # line ~70). Unpredicated CALLs are handled by HandleCall;
                # adding the CFG edge for them confuses helper-call
                # inlining and causes loops/hangs in tests that have
                # MUFU.RSQ-refinement helpers (CB_snrm2, CB_sgemm_*).
                _is_pred = (
                    block.instructions
                    and block.instructions[0].opcodes
                    and block.instructions[0].opcodes[0] == "PBRA"
                )
                if _is_pred:
                    target_addr = None
                    ops = terminator.operands or []
                    if ops:
                        target_op = ops[-1]
                        if getattr(target_op, "is_immediate", False):
                            target_addr = target_op.immediate_value
                    if target_addr is not None:
                        target_block = None
                        for k, b in block_by_addr.items():
                            if int(k, 16) == target_addr:
                                target_block = b
                                break
                        if target_block is not None and target_block not in block.succs:
                            block.add_succ(target_block)
                            target_block.add_pred(block)
            else:
                raise Exception("Unrecognized terminator")

        insert_block = {}
        for block in predicated_blocks:
            pbra_inst = block.instructions[0]
            new_block_insts = block.instructions[1:]
            # Use large address offset to avoid collision with original block
            orig_addr = int(new_block_insts[0].id, 16)
            virtual_addr = f"{orig_addr + 0x80000000:08X}"
            new_block = BasicBlock(virtual_addr, None, new_block_insts)
            block.instructions = [pbra_inst]

            new_block.pflag = block.pflag
            block.pflag = None

            insert_block[block] = new_block
            block_by_addr[new_block.addr_content] = new_block

            new_block.succs = block.succs
            block.succs = []
            for succ in new_block.succs:
                succ.preds.remove(block)
                succ.add_pred(new_block)

            new_block.add_pred(block)
            block.add_succ(new_block)

            true_br_block = new_block
            pbra_inst.operands.append(
                Operand.from_immediate(
                    true_br_block.addr_content, int(true_br_block.addr_content, 16)
                )
            )

            if block in next_block:
                false_br_block = next_block[block]
                block.add_succ(false_br_block)
                false_br_block.add_pred(block)
                pbra_inst.operands.append(
                    Operand.from_immediate(
                        false_br_block.addr_content, int(false_br_block.addr_content, 16)
                    )
                )

        old_blocks = blocks.copy()
        blocks = []
        for block in old_blocks:
            blocks.append(block)
            if block in insert_block:
                blocks.append(insert_block[block])

        for block in blocks:
            for inst in block.instructions:
                inst.parent = block

        return blocks
