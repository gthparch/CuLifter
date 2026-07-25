from transform.transform import SaSSTransform
from collections import deque

class FullDefUseAnalysis(SaSSTransform):
    def __init__(self):
        super().__init__()
        
    def apply(self, module):
        print("=== Start of FullDefUseAnalysis ===")
        
        for func in module.functions:
            self.build_def_use(func.blocks)

        print("=== End of FullDefUseAnalysis ===")
    
    def build_def_use(self, blocks):
        if not blocks:
            return

        block_gen = {}
        block_kill = {}

        for bb in blocks:
            bb.InDefs = {}
            bb.OutDefs = {}
            bb.GenDefs = {}

            gen = {}
            kill = set()

            for inst in bb.instructions:
                inst.reaching_defs = {}
                inst.users = {}

                if inst.pflag:
                    inst.pflag.defining_insts = set()
                    if inst.pflag.reg_group:
                        for sub_op in inst.pflag.reg_group:
                            sub_op.defining_insts = set()

                for op in inst.operands:
                    op.defining_insts = set()
                    if op.reg_group:
                        for sub_op in op.reg_group:
                            sub_op.defining_insts = set()

                for def_op in inst.get_defs():
                    if not def_op or not def_op.is_writable_reg:
                        continue

                    if def_op.reg_group:
                        for sub_op in def_op.reg_group:
                            if not sub_op.reg:
                                continue
                            kill.add(sub_op.reg)
                            gen[sub_op.reg] = {(inst, sub_op)}
                    else:
                        if not def_op.reg:
                            continue
                        kill.add(def_op.reg)
                        gen[def_op.reg] = {(inst, def_op)}
                        bb.GenDefs[inst] = def_op

            block_gen[bb] = gen
            block_kill[bb] = kill

        in_flow = {bb: {} for bb in blocks}
        out_flow = {bb: dict(block_gen[bb]) for bb in blocks}

        queue = deque(blocks)
        in_queue = set(blocks)

        while queue:
            bb = queue.popleft()
            in_queue.remove(bb)

            new_in = {}
            for pred_bb in bb.preds:
                pred_out = out_flow[pred_bb]
                for reg, defs in pred_out.items():
                    if reg not in new_in:
                        new_in[reg] = set(defs)
                    else:
                        new_in[reg].update(defs)

            in_flow[bb] = new_in

            new_out = dict(new_in)
            for killed_reg in block_kill[bb]:
                new_out.pop(killed_reg, None)
            new_out.update(block_gen[bb])

            curr_out = out_flow[bb]
            changed = False
            if len(new_out) != len(curr_out):
                changed = True
            else:
                for reg, defs in new_out.items():
                    if reg not in curr_out or curr_out[reg] != defs:
                        changed = True
                        break

            if changed:
                out_flow[bb] = new_out
                for succ_bb in bb.succs:
                    if succ_bb not in in_queue:
                        queue.append(succ_bb)
                        in_queue.add(succ_bb)

        for bb in blocks:
            bb.InDefs = in_flow[bb]
            bb.OutDefs = out_flow[bb]

            curr_defs = dict(bb.InDefs)

            for inst in bb.instructions:
                for use_op in inst.get_uses_with_predicate():
                    if use_op.reg_group:
                        for sub_op in use_op.reg_group:
                            if not sub_op.reg:
                                continue
                            for def_inst, def_op in curr_defs.get(sub_op.reg, set()):
                                def_inst.users.setdefault(def_op, set()).add((inst, sub_op))
                                sub_op.defining_insts.add(def_inst)
                                inst.reaching_defs[sub_op] = def_inst
                    else:
                        if not use_op.reg:
                            continue
                        for def_inst, def_op in curr_defs.get(use_op.reg, set()):
                            def_inst.users.setdefault(def_op, set()).add((inst, use_op))
                            use_op.defining_insts.add(def_inst)
                            inst.reaching_defs[use_op] = def_inst

                for def_op in inst.get_defs():
                    if def_op and def_op.is_writable_reg:
                        if def_op.reg_group:
                            for sub_op in def_op.reg_group:
                                if not sub_op.reg:
                                    continue
                                curr_defs[sub_op.reg] = {(inst, sub_op)}
                                sub_op.defining_insts = {inst}
                                inst.reaching_defs[sub_op] = inst
                        else:
                            if not def_op.reg:
                                continue
                            curr_defs[def_op.reg] = {(inst, def_op)}
                            def_op.defining_insts = {inst}
                            inst.reaching_defs[def_op] = inst


class DefUseAnalysis(SaSSTransform):
    def __init__(self):
        super().__init__()

    def apply(self, module):
        print("=== Start of DefUseAnalysis ===")

        for func in module.functions:
            self.build_def_use(func.blocks)

        print("=== End of DefUseAnalysis ===")

    def build_def_use(self, blocks):
        if not blocks:
            return

        global_defs = {}
        has_duplicate_defs = False

        for bb in blocks:
            bb.InDefs = {}
            bb.OutDefs = {}
            bb.GenDefs = {}

            for inst in bb.instructions:
                inst.reaching_defs = {}
                inst.users = {}

                if inst.pflag:
                    inst.pflag.defining_insts = set()
                    if inst.pflag.reg_group:
                        for sub_op in inst.pflag.reg_group:
                            sub_op.defining_insts = set()

                for op in inst.operands:
                    op.defining_insts = set()
                    if op.reg_group:
                        for sub_op in op.reg_group:
                            sub_op.defining_insts = set()

                for def_op in inst.get_defs():
                    if not def_op or not def_op.is_writable_reg:
                        continue

                    if def_op.reg_group:
                        for sub_op in def_op.reg_group:
                            if not sub_op.reg:
                                continue
                            if sub_op.reg in global_defs:
                                print(f"Duplicate reg {sub_op.reg} defined in SSA, previously defined in instruction {global_defs[sub_op.reg][0].id}, now re-defined in instruction {inst.id}")
                                has_duplicate_defs = True
                            global_defs[sub_op.reg] = (inst, sub_op)

                            sub_op.defining_insts = {inst}
                            inst.reaching_defs[sub_op] = inst
                    else:
                        if not def_op.reg:
                            continue
                        if def_op.reg in global_defs:
                            print(f"Duplicate reg {def_op.reg} defined in SSA, previously defined in instruction {global_defs[def_op.reg][0].id}, now re-defined in instruction {inst.id}")
                            has_duplicate_defs = True
                        global_defs[def_op.reg] = (inst, def_op)
                        bb.GenDefs[inst] = def_op

                        def_op.defining_insts = {inst}
                        inst.reaching_defs[def_op] = inst

        if has_duplicate_defs:
            raise Exception("expect SSA")

        for bb in blocks:
            for inst in bb.instructions:
                for use_op in inst.get_uses_with_predicate():
                    if use_op.reg_group:
                        for sub_op in use_op.reg_group:
                            if not sub_op.reg:
                                continue
                            def_pair = global_defs.get(sub_op.reg)
                            if not def_pair:
                                continue
                            def_inst, def_op = def_pair
                            # .get over .setdefault: setdefault allocates a
                            # throwaway set() on every hit in this hot loop
                            users = def_inst.users.get(def_op)
                            if users is None:
                                users = def_inst.users[def_op] = set()
                            users.add((inst, sub_op))
                            sub_op.defining_insts.add(def_inst)
                            inst.reaching_defs[sub_op] = def_inst
                    else:
                        if not use_op.reg:
                            continue
                        def_pair = global_defs.get(use_op.reg)
                        if not def_pair:
                            continue
                        def_inst, def_op = def_pair
                        users = def_inst.users.get(def_op)
                        if users is None:
                            users = def_inst.users[def_op] = set()
                        users.add((inst, use_op))
                        use_op.defining_insts.add(def_inst)
                        inst.reaching_defs[use_op] = def_inst
