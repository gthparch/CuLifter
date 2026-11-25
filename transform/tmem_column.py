"""Recover CUDA 13 column allocation and guarded dense MMA before SSA conversion."""
from functools import lru_cache
from pathlib import Path

from sir.instruction import Instruction
from sir.operand import Operand
from transform.tmem_pair import _compile, _match, _registers, _unknown_control_target


@lru_cache(maxsize=1)
def _aligned_patterns():
    directory = Path(__file__).with_name("patterns")
    return tuple(_compile((directory / name).read_text().splitlines()) for name in
                 ("tmem_single_aligned.sass", "tmem_single_aligned_free.sass"))


def recover_single_aligned(func):
    """Issue one collective before ptxas elects the lane that records ownership."""
    insts = func.instructions
    sites = [i for i, inst in enumerate(insts) if inst.opcodes[0] == "UTCATOMSWS"]
    if len(sites) != 3:
        return
    first, retry, free = sites
    alloc_pattern, release_pattern = _aligned_patterns()
    alloc_site = next(i for i, (line, _) in enumerate(alloc_pattern) if line.startswith("UTCATOMSWS"))
    start = first - alloc_site
    if start < 0 or insts[free].opcodes != ["UTCATOMSWS", "AND"]:
        return
    by_pc = {int(inst.id, 16): i for i, inst in enumerate(insts)}
    release_start = free + 1 - len(release_pattern)
    if release_start <= retry:
        return
    match = _match(alloc_pattern, insts, start, by_pc)
    release = _match(release_pattern, insts, release_start, by_pc)
    if not match or not release:
        return
    matched, _, regs = match
    released, _, release_regs = release
    units = regs["UNITS"]
    if (retry not in matched or units not in (1, 2, 4, 8, 16) or regs["MASK"] != (1 << units) - 1
            or any(regs[key] != release_regs[key] for key in ("MASK", "STATIC_DST", "UR10"))):
        return
    if any(by_pc.get(release_regs[label]) in matched | released for label in ("@bad_run", "@bad_start")):
        return
    # Both collectives have a single entry. Preserve the caller's branch
    # around the collective and ptxas's original release validation paths.
    for i, inst in enumerate(insts):
        if _unknown_control_target(inst):
            return
        if inst.is_branch() or inst.is_call() or inst.opcodes[0] in ("BSSY", "SSY", "PRET"):
            target = by_pc.get(inst.operands[-1].immediate_value)
            if ((i not in matched and target in matched and target != start) or
                    (i not in released and target in released and target != release_start)):
                return
    slot = insts[max(matched) - 1].operands[0]
    ncols = Operand.from_immediate(str(32 * units), 32 * units)
    insts[start] = Instruction(insts[start].id, ["TMEMALLOC1"], [slot.clone(), ncols])
    for i in (first, retry):
        site = insts[i]
        site.opcodes.append("ALLOCATED")
        site.operands[2] = slot.clone()
    insts[free].opcodes.append("ALIGNED_RELEASE")
    insts[free].operands.append(ncols.clone())


def recover_guarded_mma_constants(func):
    """Resolve uniform immediates along all incoming edges under the MMA guard."""
    insts = func.instructions
    sites = [i for i, inst in enumerate(insts) if inst.pflag and inst.opcodes[0] in ("UTCIMMA", "UTCHMMA")]
    if not sites or any(_unknown_control_target(inst) for inst in insts):
        return
    by_pc = {int(inst.id, 16): i for i, inst in enumerate(insts)}
    predecessors = [[] for _ in insts]
    for i, inst in enumerate(insts):
        if (inst.is_branch() or inst.is_call() or inst.opcodes[0] in ("BSSY", "SSY", "PRET")) and inst.operands:
            target = by_pc.get(inst.operands[-1].immediate_value)
            if target is not None:
                predecessors[target].append(i)
        terminates = ((inst.is_exit() and not inst.operands) or inst.is_return()
                      or (inst.opcodes == ["BRA"] and len(inst.operands) == 1))
        if i + 1 < len(insts) and (inst.pflag or not terminates):
            predecessors[i + 1].append(i)

    def constant(site, op, guard):
        if site == 0:  # The kernel entry has no incoming register definition.
            return
        pending = list(predecessors[site])
        seen, values = set(), set()
        while pending:
            i = pending.pop()
            if i in seen:
                continue
            seen.add(i)
            ins = insts[i]
            defs = _registers(ins.get_defs())
            if guard.reg in defs or ins.is_call():
                return
            if op.reg in defs:
                if ins.pflag and ins.pflag.reg == guard.reg and ins.pflag.is_not_reg != guard.is_not_reg:
                    pass
                elif (ins.opcodes == ["UMOV"] and (not ins.pflag or str(ins.pflag) == str(guard))
                      and ins.operands[1].is_immediate):
                    values.add(ins.operands[1].immediate_value)
                    continue
                else:
                    return
            if i == 0 or not predecessors[i]:
                return
            pending.extend(predecessors[i])
        if len(values) == 1:
            return values.pop()

    for i in sites:
        site = insts[i]
        if (len(site.operands) != 6 or not site.pflag.is_uniform_reg
                or not all(op.is_reg_group for op in site.operands[:2])
                or not site.operands[3].is_uniform_reg or not site.operands[4].is_uniform_reg):
            continue
        for j in (3, 4):
            value = constant(i, site.operands[j], site.pflag)
            if value is not None:
                site.operands[j] = Operand.from_immediate(hex(value), value)
