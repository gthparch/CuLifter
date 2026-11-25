"""Recover CUDA 13 column allocation and guarded dense MMA before SSA conversion."""
from functools import lru_cache
from pathlib import Path

from sir.instruction import Instruction
from sir.operand import Operand
from transform.tmem_pair import _compile, _match, _registers, _single_entry, _unknown_control_target


@lru_cache(maxsize=5)
def _aligned_patterns(form):
    directory = Path(__file__).with_name("patterns")
    if form == "aligned":
        names = ("tmem_single_aligned.sass", "tmem_single_aligned_free.sass")
    else:
        alloc_form = "ranked" if form == "ranked_compact" else form
        names = (f"tmem_single_{alloc_form}_alloc.sass", f"tmem_single_{form}_free.sass")
    alloc, release = [(directory / name).read_text().splitlines() for name in names]
    if form == "ranked_compact":
        # Both shifts overwrite their own mask constants once those are dead.
        substitutions = {
            "SHF.L.U32 R4, R5, R0, RZ": "SHF.L.U32 R5, R5, R0, RZ",
            "SHF.L.U32 R5, R7, R0, RZ": "SHF.L.U32 R7, R7, R0, RZ",
            "ATOMS.OR RZ, [UR7+0x14], R4": "ATOMS.OR RZ, [UR7+0x14], R5",
            "ATOMS.OR RZ, [UR7+0x18], R5": "ATOMS.OR RZ, [UR7+0x18], R7",
        }
        alloc = [substitutions.get(line, line) for line in alloc]
    return _compile(alloc), _compile(release)


def _ranked_release_slot(instructions, slot_reg, rank_reg, offset):
    # This ptxas schedule rebuilds the static slot = (CTA rank << 24) + offset
    # after allocation. Check all three reaching definitions on a straight-line
    # path to LDS. The caller rejects branches from outside into that path.
    writers = {}
    slot_ok = False
    for inst in instructions:
        if inst.is_branch() or inst.is_call() or inst.is_return():
            return False
        defs = _registers(inst.get_defs())
        if slot_reg in defs:
            slot_ok = False
            if inst.opcodes == ["ULEA"] and not inst.pflag and len(inst.operands) == 4:
                _, rank, base, shift = inst.operands
                slot_ok = (shift.immediate_value == 24 and rank.reg == rank_reg
                           and writers.get(rank_reg) == "rank" and writers.get(base.reg) == offset)
        for reg in defs:
            value = None
            if not inst.pflag and inst.opcodes == ["S2UR"] and inst.get_uses()[0].name == "SR_CgaCtaId":
                value = "rank"
            if not inst.pflag and inst.opcodes == ["UMOV"] and inst.get_uses()[0].is_immediate:
                value = inst.get_uses()[0].immediate_value
            writers[reg] = value
    return slot_ok and writers.get(rank_reg) == "rank"


def _recover_single_guarded(insts, by_pc, first, retry, free, static_bytes):
    # CUDA 12.9 checks a shared permit byte before the election. The cold
    # paths store the failure sentinel and call distinct returning trap stubs.
    alloc_pattern, free_pattern = _aligned_patterns("129")
    start = first - next(i for i, (line, _) in enumerate(alloc_pattern) if line.startswith("UTCATOMSWS"))
    # No branch labels precede the deallocation atom in this schedule.
    release = free - next(i for i, (line, _) in enumerate(free_pattern) if line.startswith("UTCATOMSWS"))
    if start < 0 or release <= retry:
        return
    alloc = _match(alloc_pattern, insts, start, by_pc)
    dealloc = _match(free_pattern, insts, release, by_pc)
    if not alloc or not dealloc:
        return
    allocated, _, regs = alloc
    released, _, free_regs = dealloc
    units, offset = regs["UNITS"], regs["STATIC_DST"]
    if (units not in (1, 2, 4, 8, 16) or regs["MASK"] != (1 << units) - 1 or
            regs["MASK"] != free_regs["MASK"] or offset < 0x400 or offset + 4 > static_bytes or
            offset >= 1 << 24 or offset % 4 or
            retry not in allocated or free not in released or allocated & released):
        return
    # The failure return and successful allocation reconverge at WARPSYNC,
    # followed by a NOP. An outer branch may bypass allocation to its end.
    end = by_pc[regs["@sync"]] + 1
    if (not _single_entry(insts, start, allocated, by_pc) or
            not _single_entry(insts, release, released, by_pc) or
            not _ranked_release_slot(insts[end+1:release], free_regs["UR4"], free_regs["UR5"], offset)):
        return
    for i, inst in enumerate(insts):
        if inst.is_call() or inst.is_return():
            if i not in allocated and i not in released:
                return
        if (inst.is_branch() or inst.is_call()) and inst.operands:
            target = by_pc.get(inst.operands[-1].immediate_value, -1)
            if end + 1 < target <= release:
                return

    elect = next(i for i in sorted(allocated) if insts[i].opcodes == ["ELECT"])
    # Keep the permit check, sentinel and election intact. On the successful
    # fallthrough, issue the collective immediately before ELECT; the original
    # elected lane then publishes its group number and updates the packed map.
    pc = int(insts[elect].id, 16) - 1
    if pc in by_pc:
        return
    slot = Operand.from_immediate(hex(offset), offset)
    ncols = Operand.from_immediate(str(32 * units), 32 * units)
    for i in (first, retry):
        insts[i].opcodes.append("ALLOCATED")
        insts[i].operands[2] = slot.clone()
    insts[free].opcodes.append("ALIGNED_RELEASE")
    insts[free].operands.append(ncols.clone())
    insts.insert(elect, Instruction(f"{pc:04x}", ["TMEMALLOC1"], [slot, ncols]))
    return True


def recover_single_aligned(func, guarded_static_bytes=0):
    """Issue one collective before ptxas elects the lane that records ownership."""
    insts = func.instructions
    sites = [i for i, inst in enumerate(insts) if inst.opcodes[0] == "UTCATOMSWS"]
    if len(sites) != 3:
        return
    first, retry, free = sites
    if insts[free].opcodes != ["UTCATOMSWS", "AND"]:
        return
    by_pc = {int(inst.id, 16): i for i, inst in enumerate(insts)}
    for form in ("aligned", "unit", "ranked", "ranked_compact"):
        alloc_pattern, release_pattern = _aligned_patterns(form)
        start = first - next(i for i, (line, _) in enumerate(alloc_pattern) if line.startswith("UTCATOMSWS"))
        release_start = free + 1 - len(release_pattern)
        if start < 0 or release_start <= retry:
            continue
        match = _match(alloc_pattern, insts, start, by_pc)
        release = _match(release_pattern, insts, release_start, by_pc)
        if match and release:
            break
    else:
        if guarded_static_bytes:
            return _recover_single_guarded(insts, by_pc, first, retry, free, guarded_static_bytes)
        return
    matched, _, regs = match
    released, _, release_regs = release
    ranked = form.startswith("ranked")
    # The one-unit protocol checks and publishes the same single bit in both maps.
    units = regs.get("UNITS", 1)
    if (retry not in matched or units not in (1, 2, 4, 8, 16) or regs.get("MASK", 1) != (1 << units) - 1
            or regs.get("MASK", 1) != release_regs.get("MASK", 1)):
        return
    if ranked:
        offset = regs["STATIC_DST"]
        if (offset >= 1 << 24 or offset % 4 or
                not _ranked_release_slot(insts[max(matched) + 1:release_start],
                                         release_regs["UR4"], release_regs["UR5"], offset)):
            return
        slot = Operand.from_immediate(hex(offset), offset)
    else:
        if any(regs[key] != release_regs[key] for key in ("STATIC_DST", "UR10")):
            return
        slot = insts[max(matched) - 1].operands[0]
    if any(by_pc.get(release_regs[label]) in matched | released for label in ("@bad_run", "@bad_start")):
        return
    # Both collectives have a single entry. Preserve the caller's branch
    # around the collective and ptxas's original release validation paths.
    for i, inst in enumerate(insts):
        if _unknown_control_target(inst):
            return
        if inst.is_branch() or inst.is_call() or inst.opcodes[0] in ("BSSY", "SSY", "PRET"):
            target = by_pc.get(inst.operands[-1].immediate_value)
            if ranked and target is not None and max(matched) + 1 < target <= release_start:
                return
            if ((i not in matched and target in matched and target != start) or
                    (i not in released and target in released and target != release_start)):
                return
    ncols = Operand.from_immediate(str(32 * units), 32 * units)
    insts[start] = Instruction(insts[start].id, ["TMEMALLOC1"], [slot.clone(), ncols])
    for i in (first, retry):
        site = insts[i]
        site.opcodes.append("ALLOCATED")
        site.operands[2] = slot.clone()
    insts[free].opcodes.append("ALIGNED_RELEASE")
    insts[free].operands.append(ncols.clone())


def recover_split_transfers(func):
    """Combine ptxas's two half-warp transfers when both always execute together."""
    insts = func.instructions
    low = "16dp32bit_t0_t15"
    high = "16dp32bit_t16_t31"
    if not any(low in inst.opcodes for inst in insts) or any(_unknown_control_target(i) for i in insts):
        return
    targets = {inst.operands[-1].immediate_value for inst in insts if inst.operands and
               (inst.is_branch() or inst.is_call() or inst.opcodes[0] in ("BSSY", "SSY", "PRET"))}
    for i, (first, last) in enumerate(zip(insts, insts[1:])):
        op = first.opcodes[0]
        if (op not in ("LDTM", "STTM") or first.opcodes[1:2] != [low] or
                last.opcodes != [op, high, *first.opcodes[2:]] or int(last.id, 16) in targets or
                first.pflag or last.pflag or len(first.operands) != 2 or len(last.operands) != 2):
            continue
        addr_index = int(op == "LDTM")
        dest, other = first.operands[1 - addr_index], last.operands[1 - addr_index]
        left, right = first.operands[addr_index], last.operands[addr_index]
        if (not dest.is_reg or dest.reg != other.reg or not left.is_mem_addr or not right.is_mem_addr or
                left.reg != right.reg or any(x.index_reg or x.suffix or x.prefix for x in (left, right))):
            continue
        offset = (right.offset or 0) - (left.offset or 0)
        if not 0 <= offset <= 0xffffffff:
            continue
        insts[i] = Instruction(first.id, [op, "SPLIT", *first.opcodes[2:]],
                               [*first.operands, Operand.from_immediate(hex(offset), offset)])
        insts[i + 1] = Instruction(last.id, ["NOP"], [])


def recover_guarded_mma_constants(func):
    """Resolve uniform immediates along all incoming edges under the MMA guard."""
    insts = func.instructions
    sites = [i for i, inst in enumerate(insts) if inst.pflag and inst.opcodes[0] in ("UTCIMMA", "UTCHMMA", "UTCQMMA")]
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
