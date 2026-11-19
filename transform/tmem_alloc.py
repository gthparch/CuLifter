"""Recognize the full-SM100-TMEM allocation and release used by dense IMMA."""


def mark_full_tmem_alloc(func):
    insts = func.instructions
    sites = [i for i, inst in enumerate(insts) if inst.opcodes[:1] == ["UTCATOMSWS"]]
    if len(sites) != 2:
        return
    start, end = sites
    alloc, free = insts[start], insts[end]
    if (alloc.opcodes != ["UTCATOMSWS", "FIND_AND_SET"] or free.opcodes != ["UTCATOMSWS", "AND"]
            or len(alloc.operands) != 3 or len(free.operands) != 2
            or alloc.operands[0].reg != "UPT" or free.operands[0].reg != "URZ"
            or not all(op.pflag and op.pflag.is_uniform_reg for op in (alloc, free))):
        return

    def writer(before, reg):
        for i in range(before - 1, -1, -1):
            inst = insts[i]
            if inst.is_branch() or inst.is_exit() or inst.is_call() or inst.is_return():
                break
            if any(reg == word.reg for dest in inst.get_defs() for word in (dest.reg_group or [dest])):
                return i, inst
        return -1, None

    for pos, site in ((start, alloc), (end, free)):
        _, vote = writer(pos, site.pflag.reg)
        if not vote or vote.opcodes not in (["VOTEU", "ANY"], ["VOTEU", "ALL"]) or vote.pflag:
            return
    _, count = writer(start, alloc.operands[2].reg)
    if (not count or count.opcodes != ["UMOV"] or count.pflag
            or count.get_uses()[0].immediate_value != 16):
        return
    # A general AND changes arbitrary SW-state bits. Only this release mask
    # frees one full allocation: ~((0x1ffff) << (taddr >> 5)).
    pos, invert = writer(end, free.operands[1].reg)
    if not invert or invert.opcodes != ["ULOP3", "LUT"] or str(invert.pflag) != str(free.pflag):
        return
    a, bits, c, lut, pred = invert.get_uses()
    if not (a.is_rz and c.is_rz and lut.immediate_value == 0x33 and pred.is_pt and pred.is_not_reg):
        return
    pos, uniform = writer(pos, bits.reg)
    if not uniform or uniform.opcodes != ["R2UR"]:
        return
    if uniform.pflag:
        _, vote = writer(end, free.pflag.reg)
        expected = ["VOTEU", "ALL" if uniform.pflag.is_not_reg else "ANY"]
        if (vote.opcodes != expected or vote.get_uses()[0].reg != uniform.pflag.reg
                or bool(free.pflag.is_not_reg) != bool(uniform.pflag.is_not_reg)):
            return
    pos, shift = writer(pos, uniform.get_uses()[0].reg)
    if (not shift or shift.opcodes != ["SHF", "L", "U32"]
            or str(shift.pflag) != str(uniform.pflag)):
        return
    ones, index, zero = shift.get_uses()
    _, mask = writer(pos, ones.reg)
    _, group = writer(pos, index.reg)
    if (not zero.is_rz or not mask or mask.opcodes != ["IMAD", "MOV", "U32"]
            or not all(src.is_rz for src in mask.get_uses()[:2])
            or mask.get_uses()[2].immediate_value != 0x1ffff
            or not group or group.opcodes != ["SHF", "R", "U32", "HI"]
            or not group.get_uses()[0].is_rz or group.get_uses()[1].immediate_value != 5
            or not all(str(inst.pflag) == str(uniform.pflag) for inst in (mask, group))):
        return
    alloc.opcodes.append("FULL_TMEM")
    free.opcodes.append("FULL_TMEM")
