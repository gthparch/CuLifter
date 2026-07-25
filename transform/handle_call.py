from collections import deque
from transform.transform import SaSSTransform
from sir.basicblock import BasicBlock
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import Operand


class HandleCall(SaSSTransform):
    INLINE_THRESHOLD = 16

    def apply(self, module):
        print("=== Start of HandleCall ===")
        elf = getattr(module, "elf", None)
        for func in list(module.functions):
            self.normalize_calls(func, elf)
            self.handle_calls(module, func)
        print("=== End of HandleCall ===")

    def normalize_calls(self, func, elf):
        if elf is None:
            return
        self.lower_abs_calls_to_callfunc(func, elf)

    def lower_abs_calls_to_callfunc(self, func, elf):
        rel_map = elf.rel_text_call_targets(func.name)
        if not rel_map or not func.blocks:
            return

        for block in func.blocks:
            for idx, inst in enumerate(list(block.instructions)):
                if not inst.opcodes or inst.opcodes[0] != "CALL":
                    continue
                if "ABS" not in inst.opcodes or "NOINC" not in inst.opcodes:
                    continue

                uses = inst.get_uses()
                if len(uses) != 1:
                    continue
                target_op = uses[0]
                if not target_op.is_immediate or int(target_op.immediate_value or 0) != 0:
                    continue

                call_site = int(inst.id, 16)
                callee = rel_map.get(call_site)
                if not callee:
                    continue

                new_inst = Instruction(
                    id=inst.id,
                    opcodes=["CALLFUNC"],
                    operands=[Operand.from_symbol(callee)],
                    parentBB=block,
                    pflag=inst.pflag,
                )
                new_inst.inst_content = f"CALLFUNC {callee}"
                new_inst.ctl_code = inst.ctl_code
                new_inst.true_branch = inst.true_branch
                new_inst.false_branch = inst.false_branch

                block.instructions[idx] = new_inst

                # If CALL was a block terminator, preserve fallthrough by inserting a BRA.
                if idx == len(block.instructions) - 1 and block.succs:
                    succ = block.succs[0]
                    dest_op = Operand.from_immediate(succ.addr_content, succ.addr)
                    bra_inst = Instruction(
                        id=f"{block.addr_content}_syn_callfunc",
                        opcodes=["BRA"],
                        operands=[dest_op],
                        parentBB=block,
                        pflag=None,
                    )
                    block.instructions.append(bra_inst)

    def handle_calls(self, module, func):
        if not func.blocks:
            return

        all_blocks = list(func.blocks)
        device_funcs, main_blocks = self.extract_device_functions(func.blocks)
        if not device_funcs:
            return

        call_usage = self.collect_call_usage(all_blocks, device_funcs)

        # Cyclicity split: ACYCLIC call-only subroutines (their call graph is
        # a DAG) are ALWAYS inlined at every site — no count threshold, no
        # per-target cap. The SASS CALL ABI (args/returns in caller registers,
        # return address in a register) is not modeled, so an extracted
        # Function lifts with no dataflow into/out of it — mr-cuda's u64-div
        # helper has 17 call sites (> INLINE_THRESHOLD), was extracted, and
        # produced garbage for every input. Inlining a DAG always terminates,
        # so the runaway-expansion caps are unnecessary there. Targets that
        # can REACH a cycle (direct or mutual recursion — e.g. frechet-cuda's
        # distance_norm3 chain) keep the original threshold + cap behavior,
        # which is what bounds their expansion.
        cycle_reaching = self.cycle_reaching_funcs(device_funcs)

        extracted = self.extract_call_funcs(
            module, func, device_funcs, call_usage, cycle_reaching)
        if extracted:
            self.replace_calls_with_callfunc(all_blocks, extracted)

        func.blocks = main_blocks

        inline_targets = {
            addr
            for addr, count in call_usage.items()
            if count > 0 and addr in device_funcs
            and (addr not in cycle_reaching or count <= self.INLINE_THRESHOLD)
        }

        self.inline_calls(func, device_funcs, inline_targets, cycle_reaching)
        for new_func in extracted.values():
            self.inline_calls(new_func, device_funcs, inline_targets,
                              cycle_reaching)

    def build_call_graph(self, device_funcs):
        """addr -> set of direct-call target addrs that are device funcs."""
        edges = {}
        for addr, blocks in device_funcs.items():
            callees = set()
            for block in blocks:
                term = block.get_terminator()
                if term is None or not term.is_call():
                    continue
                target = self.direct_call_target(term)
                if target is not None and target in device_funcs:
                    callees.add(target)
            edges[addr] = callees
        return edges

    def cycle_reaching_funcs(self, device_funcs):
        """Addrs of device functions whose call graph can reach a cycle
        (self-recursion or mutual recursion, directly or transitively).
        Inlining such a function replicates the recursive chain's call
        sites, so only the threshold/cap machinery bounds its expansion;
        everything else forms a DAG and inlining it always terminates."""
        edges = self.build_call_graph(device_funcs)
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {addr: WHITE for addr in edges}
        reaches = {addr: False for addr in edges}

        # Iterative DFS (recursion depth is unbounded by Python's limit on
        # pathological call graphs). A GRAY successor is on the current DFS
        # stack → back edge → cycle; a finished (BLACK) successor propagates
        # its memoized answer.
        for root in edges:
            if color[root] != WHITE:
                continue
            stack = [(root, iter(edges[root]))]
            color[root] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for succ in it:
                    if color[succ] == GRAY:
                        reaches[node] = True
                    elif color[succ] == WHITE:
                        color[succ] = GRAY
                        stack.append((succ, iter(edges[succ])))
                        advanced = True
                        break
                    elif reaches[succ]:
                        reaches[node] = True
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
                    if reaches[node] and stack:
                        reaches[stack[-1][0]] = True
        return {addr for addr, r in reaches.items() if r}

    def collect_call_usage(self, blocks, device_funcs):
        usage = {}
        targets = set(device_funcs)

        for block in blocks:
            term = block.get_terminator()
            if term is None or not term.is_call():
                continue

            target_addr = self.direct_call_target(term)
            if target_addr is None or target_addr not in targets:
                continue
            usage[target_addr] = usage.get(target_addr, 0) + 1

        return usage

    def extract_call_funcs(self, module, parent_func, device_funcs, call_usage,
                           cycle_reaching=frozenset()):
        extracted = {}
        for target_addr, count in call_usage.items():
            if count <= self.INLINE_THRESHOLD:
                continue
            if target_addr not in cycle_reaching:
                # Acyclic target: always inlined at every call site instead
                # (extraction breaks the unmodeled SASS call ABI — the
                # extracted Function lifts with no argument/return dataflow).
                continue
            if target_addr not in device_funcs:
                continue

            blocks = device_funcs[target_addr]
            if not blocks:
                continue

            base_name = f"{parent_func.name}_call_{blocks[0].addr_content}"
            name = self.unique_function_name(module, base_name)
            new_func = Function(name)
            new_func.blocks = blocks
            module.functions.append(new_func)
            extracted[target_addr] = new_func

        return extracted

    def direct_call_target(self, inst):
        if not inst.operands:
            return None
        op = inst.operands[0]
        if not getattr(op, "is_immediate", False):
            return None
        return op.immediate_value

    def replace_calls_with_callfunc(self, blocks, extracted):
        for block in blocks:
            term = block.get_terminator()
            if term is None or not term.is_call():
                continue

            target_addr = self.direct_call_target(term)
            callee = extracted.get(target_addr)
            if callee is None:
                continue

            call_idx = block.instructions.index(term)
            self.replace_call_with_callfunc(block, call_idx, term, callee.name)

    def replace_call_with_callfunc(self, block, call_idx, call_inst, callee_name):
        new_inst = Instruction(
            id=call_inst.id,
            opcodes=["CALLFUNC"],
            operands=[Operand.from_symbol(callee_name)],
            parentBB=block,
            pflag=call_inst.pflag,
        )
        new_inst.inst_content = f"CALLFUNC {callee_name}"
        new_inst.ctl_code = call_inst.ctl_code
        new_inst.true_branch = call_inst.true_branch
        new_inst.false_branch = call_inst.false_branch

        block.instructions[call_idx] = new_inst

        if call_idx != len(block.instructions) - 1 or not block.succs:
            return

        succ = block.succs[0]
        dest_op = Operand.from_immediate(succ.addr_content, succ.addr)
        bra_id_base = f"{block.addr_content}_syn_callfunc"
        bra_id = bra_id_base
        existing_ids = {inst.id for inst in block.instructions}
        dedup = 0
        while bra_id in existing_ids:
            dedup += 1
            bra_id = f"{bra_id_base}_{dedup:08x}"
        bra_inst = Instruction(
            id=bra_id,
            opcodes=["BRA"],
            operands=[dest_op],
            parentBB=block,
            pflag=None,
        )
        block.instructions.append(bra_inst)

    # Safety bound for CYCLE-REACHING targets only: each such target can be
    # inlined at most this many times across the whole function. Without
    # this, a device function that (directly or via mutual recursion) calls
    # itself will loop forever as each inline clones blocks whose calls the
    # outer while-loop rediscovers. Frechet-cuda triggers this via
    # distance_norm3 which recursively chains into a helper that ends up
    # calling itself. Observed as a 99% CPU hang. Set to INLINE_THRESHOLD
    # (the original per-function call-count limit); higher values just grow
    # the inlined code without helping correctness. Acyclic (DAG) targets
    # are exempt — their expansion is finite by construction, and capping
    # them leaves raw CALLs whose ABI the lifter cannot model (mr-cuda).
    MAX_INLINE_PER_TARGET = INLINE_THRESHOLD

    def inline_calls(self, func, device_funcs, inline_targets,
                     cycle_reaching=frozenset()):
        inline_counts = {addr: 0 for addr in inline_targets}
        changed = True
        iterations = 0
        # Hard ceiling in case accounting misses a target (belt + suspenders).
        # Typical kernels need <20 iterations; 4096 is 64 targets × 64 expansions.
        max_iterations = 4096
        while changed:
            changed = False
            iterations += 1
            if iterations > max_iterations:
                print(f"[HandleCall] inline_calls: hit global iteration cap "
                      f"({max_iterations}) in {func.name}; leaving remaining "
                      f"calls as CALLFUNC")
                break
            for block in func.blocks:
                terminator = block.get_terminator()
                if terminator is None or not terminator.is_call():
                    continue

                target_addr = self.direct_call_target(terminator)
                if target_addr is None:
                    continue
                if target_addr not in inline_targets:
                    continue
                if target_addr not in device_funcs:
                    continue

                if (target_addr in cycle_reaching
                        and inline_counts[target_addr] >= self.MAX_INLINE_PER_TARGET):
                    # Cycle-reaching target already inlined up to the
                    # per-target cap — skip this site; it'll remain a
                    # CALLFUNC (recursion boundary).
                    continue

                self.inline_call(func, block, device_funcs[target_addr])
                inline_counts[target_addr] += 1
                changed = True
                break

    def unique_function_name(self, module, base_name):
        existing = {func.name for func in module.functions}
        if base_name not in existing:
            return base_name

        dedup = 0
        name = base_name
        while name in existing:
            dedup += 1
            name = f"{base_name}_{dedup:08x}"
        return name

    def extract_device_functions(self, blocks):
        visited = set()
        device_funcs = {}

        main_blocks = self.bfs_cfg(blocks[0], visited)

        for block in blocks:
            if block.addr not in visited:
                cfg_blocks = self.bfs_cfg(block, visited)
                if cfg_blocks:
                    first_addr = cfg_blocks[0].addr
                    device_funcs[first_addr] = cfg_blocks

        return device_funcs, main_blocks

    def bfs_cfg(self, start_block, visited):
        if start_block.addr in visited:
            return []

        result = []
        queue = deque([start_block])
        local_visited = set()

        while queue:
            block = queue.popleft()
            if block.addr in local_visited:
                continue
            local_visited.add(block.addr)
            visited.add(block.addr)
            result.append(block)

            for succ in block.succs:
                if succ.addr not in local_visited:
                    queue.append(succ)

            for pred in block.preds:
                if pred.addr not in local_visited:
                    queue.append(pred)

        result.sort(key=lambda b: b.addr)
        return result

    def create_bra(self, inst_id, target_block):
        dest_op = Operand.from_immediate(target_block.addr_content, target_block.addr)
        bra_inst = Instruction(id=inst_id, opcodes=["BRA"], operands=[dest_op])
        return bra_inst

    def inline_call(self, func, call_block, device_blocks):
        call_inst = call_block.get_terminator()
        fallthrough_succs = call_block.succs[:]
        inline_prefix = str(call_inst.id).lower()
        existing_inst_ids = {
            inst.id for bb in func.blocks for inst in getattr(bb, "instructions", [])
        }
        existing_block_ids = {bb.addr_content for bb in func.blocks}

        cloned_blocks = []
        old_to_new = {}

        for orig_block in device_blocks:
            new_insts = []
            for inst in orig_block.instructions:
                cloned_inst = inst.clone()
                base_id = f"{inline_prefix}_{str(cloned_inst.id).lower()}"
                new_id = base_id
                dedup = 0
                while new_id in existing_inst_ids:
                    dedup += 1
                    new_id = f"{base_id}_{dedup:08x}"
                existing_inst_ids.add(new_id)
                cloned_inst.id = new_id
                new_insts.append(cloned_inst)

            base_block_id = f"{inline_prefix}_{orig_block.addr_content.lower()}"
            new_block_id = base_block_id
            dedup = 0
            while new_block_id in existing_block_ids:
                dedup += 1
                new_block_id = f"{base_block_id}_{dedup:08x}"
            existing_block_ids.add(new_block_id)

            new_block = BasicBlock(new_block_id, orig_block.pflag, new_insts)
            for inst in new_block.instructions:
                inst.parent = new_block
            old_to_new[orig_block.addr] = new_block
            cloned_blocks.append(new_block)

        for orig_block in device_blocks:
            new_block = old_to_new[orig_block.addr]
            for pred in orig_block.preds:
                if pred.addr in old_to_new:
                    new_block.preds.append(old_to_new[pred.addr])
            for succ in orig_block.succs:
                if succ.addr in old_to_new:
                    new_block.succs.append(old_to_new[succ.addr])

        # Rewrite internal branch operands to target the cloned blocks' virtual addresses.
        for block in cloned_blocks:
            for inst in block.instructions:
                if not inst.is_branch() or inst.is_indirect_branch():
                    continue

                # PBRA: [pred, true, false]
                if inst.opcodes and inst.opcodes[0] == "PBRA":
                    target_ops = []
                    if len(inst.operands) > 1:
                        target_ops.append(inst.operands[1])
                    if len(inst.operands) > 2:
                        target_ops.append(inst.operands[2])
                else:
                    target_ops = [inst.operands[-1]] if inst.operands else []

                for op in target_ops:
                    if not getattr(op, "is_immediate", False) or op.immediate_value is None:
                        continue
                    new_target = old_to_new.get(op.immediate_value)
                    if not new_target:
                        continue
                    op.name = new_target.addr_content
                    op.immediate_value = new_target.addr

        entry_block = cloned_blocks[0]

        # Replace CALL with BRA
        call_idx = call_block.instructions.index(call_inst)
        bra_to_entry = self.create_bra(call_inst.id, entry_block)
        bra_to_entry.parent = call_block
        call_block.instructions[call_idx] = bra_to_entry

        for succ in fallthrough_succs:
            if call_block in succ.preds:
                succ.preds.remove(call_block)
        call_block.succs = [entry_block]
        entry_block.preds.append(call_block)

        # Replace RET with BRA
        for block in cloned_blocks:
            terminator = block.get_terminator()
            if terminator and terminator.is_return():
                ret_idx = block.instructions.index(terminator)
                if fallthrough_succs:
                    bra_to_cont = self.create_bra(terminator.id, fallthrough_succs[0])
                    bra_to_cont.parent = block
                    block.instructions[ret_idx] = bra_to_cont
                block.succs = fallthrough_succs[:]
                for succ in fallthrough_succs:
                    succ.preds.append(block)

        insert_idx = func.blocks.index(call_block) + 1
        func.blocks = func.blocks[:insert_idx] + cloned_blocks + func.blocks[insert_idx:]
