from transform.transform import SaSSTransform


class RecordDescriptorConsts(SaSSTransform):
    """Record descriptor-register const loads BEFORE SSA/RegRemap rename them.

    SM90 descriptor-addressed memory ops (`LDG.E/STG.E/ATOMG desc[URx][Ry.64]`)
    name the descriptor register only in the address operand's `desc_reg`
    string (e.g. "UR14"), never as a real reg_group member. `desc_reg` is set
    at parse time and is NOT renamed by SSA/RegRemap (those only touch
    `op.reg`/reg_group). So by lift time `desc_reg` still says "UR14" while the
    descriptor-defining `ULDC.64 UR14, c[0x0][0x208]` has been renamed (and,
    because liveness can't see the desc_reg use, usually DCE'd).

    `_resolve_descriptor_addr` therefore can't find the descriptor base in
    IRRegs and silently skips, leaving the raw register offset (a cute Layout
    value) to be dereferenced as a pointer -> SIGSEGV (the FA3 desc[UR14] crash).

    This pass runs before SSA, while register names are still the original
    parse-time names that `desc_reg` references. It records, per function,
    {original_reg_name -> (const_bank, const_offset)} for every uniform/scalar
    const load (`ULDC`/`LDC`). The lifter consults this map as a fallback: when
    IRRegs has no value for `desc_reg`, it materializes the descriptor base
    directly from const memory at the recorded offset — needing neither the
    renamed register nor the (DCE'd) defining instruction.
    """

    def apply(self, module):
        super().apply(module)
        for func in module.functions:
            state = {}
            assignments = {}

            def reg_names(op):
                if getattr(op, "reg", None):
                    return [op.reg]
                if getattr(op, "is_reg_group", False) and op.reg_group:
                    return [part.reg for part in op.reg_group if part.reg]
                return []

            for block in func.blocks:
                for inst in block.instructions:
                    if not inst.opcodes:
                        continue

                    # Attach the reaching descriptor definition before SSA.
                    # desc_reg is not a normal use and will retain its original
                    # spelling even when the defining register is renamed.
                    for operand in inst.operands:
                        desc_reg = getattr(operand, "desc_reg", None)
                        if desc_reg and desc_reg in state:
                            operand.desc_const = state[desc_reg]

                    defs = inst.get_defs()
                    uses = inst.get_uses()

                    copy_value = None
                    if inst.opcodes[0] in ("R2UR", "MOV", "UMOV") and uses:
                        src_regs = reg_names(uses[0])
                        if src_regs:
                            copy_value = state.get(src_regs[0])

                    for dst in defs:
                        for reg in reg_names(dst):
                            state.pop(reg, None)

                    if inst.opcodes[0] in ("ULDC", "LDC", "LDCU") and defs and uses:
                        src = uses[0]
                        if getattr(src, "is_const_mem", False) \
                                and src.offset_value is not None:
                            dst_regs = reg_names(defs[0])
                            if dst_regs:
                                value = (
                                    src.const_mem_bank
                                    if src.const_mem_bank is not None else 0,
                                    src.offset_value)
                                state[dst_regs[0]] = value
                                assignments.setdefault(
                                    dst_regs[0], set()).add(value)
                    elif copy_value is not None and defs:
                        dst_regs = reg_names(defs[0])
                        if dst_regs:
                            state[dst_regs[0]] = copy_value
                            assignments.setdefault(
                                dst_regs[0], set()).add(copy_value)

            # Function-wide fallback for uses that could not be annotated.
            # Per-use annotations handle registers reloaded with different
            # descriptor bases.
            func.desc_const_map = {
                reg: next(iter(values))
                for reg, values in assignments.items()
                if len(values) == 1
            }
