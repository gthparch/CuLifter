from transform.transform import SaSSTransform


class RegRemap(SaSSTransform):
    def apply(self, module):
        print("=== Start of RegRemap ===")
        for func in module.functions:
            if not func.blocks:
                continue
            self.remap_names(func)
        print("=== End of RegRemap ===")

    def remap_names(self, function):
        r_name_map = {}
        p_name_map = {}
        r_counter = 1
        p_counter = 1

        r_names = set()
        p_names = set()
        for block in function.blocks:
            for inst in block.instructions:
                for op in inst.operands:
                    if not op.is_writable_reg:
                        continue
                    if op.is_reg_group:
                        for sub_op in op.reg_group:
                            if not sub_op.reg:
                                continue
                            if not sub_op.is_writable_reg:
                                continue
                            if sub_op.is_predicate_reg:
                                p_names.add(sub_op.reg)
                            else:
                                r_names.add(sub_op.reg)
                    else:
                        if not op.reg:
                            continue
                        if op.is_predicate_reg:
                            p_names.add(op.reg)
                        else:
                            r_names.add(op.reg)

        for name in sorted(r_names):
            r_name_map[name] = f"R{r_counter}"
            r_counter += 1
        for name in sorted(p_names):
            p_name_map[name] = f"P{p_counter}"
            p_counter += 1

        for block in function.blocks:
            for inst in block.instructions:
                if inst.predicated() and inst.pflag.reg in p_name_map:
                    inst.pflag.set_reg(p_name_map[inst.pflag.reg])
                for op in inst.operands:
                    if op.is_reg_group:
                        for sub_op in op.reg_group:
                            if sub_op.is_predicate_reg and sub_op.reg in p_name_map:
                                sub_op.set_reg(p_name_map[sub_op.reg])
                            elif sub_op.reg in r_name_map:
                                sub_op.set_reg(r_name_map[sub_op.reg])
                    else:
                        if op.is_predicate_reg and op.reg in p_name_map:
                            op.set_reg(p_name_map[op.reg])
                        elif op.reg in r_name_map:
                            op.set_reg(r_name_map[op.reg])