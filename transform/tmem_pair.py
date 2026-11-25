"""Recover paired TMEM operations before the CFG splits ptxas's elected-lane paths.

The patterns describe the complete SASS protocols for a parameter or static
shared slot and a one-unit, multi-unit or runtime-sized allocation. Match the
instruction operands, register relationships, branch targets and cold wait
paths. Reissue the PTX collective at the converged entry and preserve the
consumer's loads/stores. Unrecognized SW-state protocols go to the existing
unsupported-instruction handler.
"""
import re
from functools import lru_cache
from pathlib import Path

from sir.instruction import Instruction
from sir.operand import Operand


_REGISTER = re.compile(r"(U?[RP])\d+")
_TOKEN = re.compile(r"@[a-z_][a-z_0-9]*|\w+|[^\w\s]")


def _tokens(line):
    tokens = _TOKEN.findall(line.replace(".reuse", ""))
    # ptxas uses both MOV and the zero-product IMAD to copy a uniform
    # register into a general register during allocation/release.
    if (len(tokens) == 10 and tokens[:3] == ["IMAD", ".", "U32"] and
            tokens[4:9] == [",", "RZ", ",", "RZ", ","] and
            re.fullmatch(r"R\d+", tokens[3]) and re.fullmatch(r"UR\d+", tokens[9])):
        return ["MOV", tokens[3], ",", tokens[9]]
    return tokens


def _read_lines(form):
    path = Path(__file__).with_name("patterns") / f"tmem_pair_{form}.sass"
    return [line for line in path.read_text().splitlines() if line and not line.startswith("#")]


def _compile(lines):
    return [(line, _tokens(line.removeprefix("KEEP "))) for line in lines]


@lru_cache(maxsize=1)
def _patterns():
    patterns = []
    for form in ("single", "multi", "dynamic", "static_single", "static_multi", "static_dynamic"):
        lines = _read_lines(form)
        patterns.append(lines)
        if form in ("single", "multi", "dynamic"):
            # With a following global store ptxas schedules a descriptor load
            # at the converged odd-CTA election. Keep it for the consumer.
            middle = lines.index("odd_elect:") + 2
            observed = lines.copy()
            parity, election = lines[middle:middle + 2]
            observed[middle:middle + 2] = [election, parity, "KEEP LDCU.64 UR99, c[0x0][0x358]"]
            patterns.append(observed)
    for form in ("single", "multi", "dynamic", "static_single", "static_multi", "static_dynamic"):
        lines = _read_lines("129_" + form)
        patterns.append(lines)
        if form in ("single", "multi"):
            # The compiler may schedule a consumer's descriptor load before
            # the guardrail test. Keep the value live across the collective.
            observed = lines.copy()
            observed.insert(3, "KEEP LDCU.64 UR99, c[0x0][0x358]")
            patterns.append(observed)
    return [_compile(lines) for lines in patterns]


@lru_cache(maxsize=2)
def _release_pattern(single):
    return _compile(_read_lines("free_single" if single else "free_multi"))


@lru_cache(maxsize=2)
def _release_129(single):
    return _compile(_read_lines("129_free_single" if single else "129_free_multi"))


def _source(inst):
    if inst.inst_content is None:
        return ""
    body = inst.inst_content
    if inst.is_return():
        # cuobjdump prints a space between the return register and offset,
        # which the parsed raw operand string elides.
        body = re.sub(r"(R\d+)(0x0)$", r"\1 \2", body)
    return (("@" + str(inst.pflag) + " ") if inst.pflag else "") + body


def _bind(bindings, name, value):
    return bindings.setdefault(name, value) == value


def _match_tokens(pattern, actual, bindings, reverse):
    if len(pattern) != len(actual):
        return False
    for name, value in zip(pattern, actual):
        register = _REGISTER.fullmatch(name)
        if register:
            if not re.fullmatch(register[1] + r"\d+", value) or not _bind(reverse, value, name):
                return False
        elif (name.startswith("@") and name != "@") or name in ("DST", "STATIC_BASE", "STATIC_DST", "NCOLS", "UNITS", "MASK"):
            if not re.fullmatch("0x[0-9a-fA-F]+", value):
                return False
            value = int(value, 16)
        elif name == value:
            continue
        else:
            return False
        if not _bind(bindings, name, value):
            return False
    return True


def _match(pattern, instructions, start, by_pc):
    bindings, reverse, matched, keep = {}, {}, set(), set()
    index = start
    for line, tokens in pattern:
        if line.endswith(":"):
            seek = line.startswith("seek ")
            label = "@" + line.removeprefix("seek ")[:-1]
            if label == "@odd_elect":
                # The odd CTA reloads its own destination after reconvergence.
                # The even CTA's uniform allocation temporaries are now dead.
                for reg in list(bindings):
                    if reg.startswith("UR"):
                        reverse.pop(bindings.pop(reg), None)
            if seek or label in ("@even_slow", "@odd_slow", "@trap_start", "@trap_mask"):
                index = by_pc.get(bindings[label], -1)
            if not 0 <= index < len(instructions) or not _bind(bindings, label, int(instructions[index].id, 16)):
                return
            continue
        if index in matched or not 0 <= index < len(instructions):
            return
        if not _match_tokens(tokens, _tokens(_source(instructions[index])), bindings, reverse):
            return
        # STAS consumes both words of the .64 address, including the
        # implicitly adjacent high register, even under register renaming.
        if line.startswith("STAS"):
            reg = tokens[tokens.index("[") + 1]
            high = "R" + str(int(reg[1:]) + 1)
            if not _bind(bindings, high, "R" + str(int(bindings[reg][1:]) + 1)):
                return
        matched.add(index)
        if line.startswith("KEEP "):
            keep.add(index)
        index += 1

    return matched, keep, bindings


def _alloc_operands(bindings):
    units = bindings.get("UNITS", 1)
    if units not in (1, 2, 4, 8, 16) or bindings.get("MASK", (1 << units) - 1) != (1 << units) - 1:
        return
    columns = (Operand.parse(f"c[0x0][{bindings['NCOLS']:#x}]") if "NCOLS" in bindings
               else Operand.from_immediate(str(32 * units), 32 * units))
    if "STATIC_DST" in bindings:
        offset = bindings["STATIC_DST"]
        if offset % 4 or offset >= 1 << 24:
            return
        dst = Operand.from_immediate(str(offset), offset)
    else:
        dst = Operand.parse(f"c[0x0][{bindings['DST']:#x}]")
    return dst, columns


def _registers(operands):
    regs = set()
    for op in operands:
        regs.update(reg for reg in (op.reg, op.index_reg, op.desc_reg) if reg and _REGISTER.fullmatch(reg))
        regs.update(word.reg for word in op.reg_group or [] if _REGISTER.fullmatch(word.reg or ""))
        if op.desc_reg and _REGISTER.fullmatch(op.desc_reg):
            regs.add("UR" + str(int(op.desc_reg[2:]) + 1))
    return regs


def _unknown_control_target(inst):
    if inst.opcodes[0] in ("BRX", "BRXU", "JMP", "JMX", "JMXU", "SWITCH"):
        return True
    if inst.is_call():
        return len(inst.operands) != 1 or not inst.operands[0].is_immediate
    return False


def _single_entry(instructions, start, matched, by_pc):
    # A branch into either retry/hand-off path would bypass the collective.
    for i, inst in enumerate(instructions):
        if i in matched:
            continue
        if _unknown_control_target(inst) or inst.opcodes[0] in ("BSSY", "SSY", "PRET"):
            return False
        if inst.is_branch() or inst.is_call():
            target = inst.operands[-1].immediate_value
            if target not in by_pc or (by_pc[target] in matched and by_pc[target] != start):
                return False
        if i + 1 in matched and i + 1 != start:
            jump = inst.opcodes == ["BRA"] and len(inst.operands) == 1
            terminal = (inst.is_exit() and not inst.operands) or inst.is_return() or jump
            if inst.pflag or not terminal:
                return False

    return True


_CONSUMER_OPS = {"LDC", "LDCU", "LDS", "STG", "STS", "MOV", "IMAD", "NOP", "S2R", "S2UR", "LEA", "ULEA"}


def _scratch_dead(instructions, matched, keep, join, release):
    # Compiler temporaries inside the protocol must be dead at its exit.
    # The descriptor at KEEP is defined on all paths through the region.
    scratch = set().union(*(_registers(instructions[i].get_defs()) for i in matched - keep))
    for i in keep:
        saved = _registers(instructions[i].get_defs())
        if any(saved & _registers(instructions[j].get_defs()) for j in matched if i < j <= join):
            return False
        scratch -= saved
    for i in range(join + 1, len(instructions)):
        if release and i == release[0]:
            return release[1].reg not in scratch
        inst = instructions[i]
        if inst.is_exit() and not inst.pflag and not inst.operands:
            return True
        if inst.is_branch() or inst.is_return() or inst.is_call():
            return not scratch
        if inst.opcodes[0] not in _CONSUMER_OPS:
            return not scratch
        uses = _registers(inst.get_uses() + ([inst.pflag] if inst.pflag else []))
        if scratch & uses:
            return False
        if not inst.pflag:
            scratch -= _registers(inst.get_defs())
    return False


def _paired_release(insts, join, by_pc, bindings, dst, columns):
    # Recover the matching dealloc/relinquish only when the caller reads
    # back this unmodified allocation and ptxas checks both ownership masks.
    if not columns.is_immediate or not dst.is_const_mem:
        return
    guardrail = "@alloc_trap" in bindings
    writers = {}
    for i in range(join + 1, len(insts)):
        inst = insts[i]
        if guardrail and inst.opcodes == ["UMOV"] and inst.get_uses()[0].immediate_value == 0x50:
            break
        if not guardrail and inst.opcodes == ["LDS"] and inst.get_uses() and inst.get_uses()[0].offset == 0x14:
            break
        if inst.pflag or inst.opcodes[0] not in _CONSUMER_OPS - {"STS"}:
            return
        for reg in _registers(inst.get_defs()):
            writers[reg] = i
    else:
        return

    ncols = columns.immediate_value
    pattern = (_release_129(ncols == 32) if guardrail else _release_pattern(ncols == 32))
    result = _match(pattern, insts, i, by_pc)
    if result is None:
        return
    sites, keep, regs = result
    if guardrail:
        if ncols != 32:
            marker = writers.get(regs["R11"])
            if marker is None or _tokens(_source(insts[marker])) != _tokens(
                    f"IMAD.MOV.U32 {regs['R11']}, RZ, RZ, 0x1"):
                return
        rank_writer = writers.get(regs["UR5"])
        if rank_writer is None or _tokens(_source(insts[rank_writer])) != _tokens("S2UR " + regs["UR5"] + ",SR_CgaCtaId"):
            return
    elif regs["R0"] != bindings["R0"] or regs["R0"] in writers:
        return
    mask_register = "R7" if guardrail and ncols == 32 else "R9"
    mask_index, address_index = writers.get(regs[mask_register]), writers.get(regs["R5"])
    if mask_index is None or address_index is None:
        return
    mask, address = insts[mask_index], insts[address_index]
    if (mask.opcodes != ["IMAD", "MOV", "U32"] or not all(op.is_rz for op in mask.get_uses()[:2])
            or mask.get_uses()[-1].immediate_value != (1 << (ncols // 32)) - 1 or address.opcodes != ["LDS"]):
        return
    slot = address.get_uses()[0]
    if not slot.is_mem_addr or slot.offset or slot.index_reg or slot.suffix:
        return
    for producer in reversed(insts[join + 1:address_index]):
        if slot.reg in _registers(producer.get_defs()):
            if producer.opcodes != ["LDCU"] or str(producer.get_uses()[0]) != str(dst):
                return
            return i, Operand.parse(regs["R5"]), sites, keep


@lru_cache(maxsize=1)
def _ranked_patterns():
    return tuple(_compile(_read_lines(name)) for name in ("ranked_multi", "ranked_free"))


def _ranked_edges(insts, by_pc):
    edges = []
    for i, inst in enumerate(insts):
        if _unknown_control_target(inst):
            return
        nexts = []
        if inst.is_branch() or inst.is_call() or inst.opcodes[0] in ("BSSY", "SSY", "PRET"):
            target = by_pc.get(inst.operands[-1].immediate_value)
            if target is None:
                return
            nexts.append(target)
        terminal = (inst.is_exit() and not inst.operands) or inst.is_return() or (inst.opcodes == ["BRA"] and len(inst.operands) == 1)
        if (inst.pflag or not terminal) and i + 1 < len(insts):
            nexts.append(i + 1)
        edges.append(nexts)
    return edges


def _ranked_base(insts, edges, start, released, regs):
    # ptxas hoists (CTA rank << 24) + static shared base before the first
    # branch. Both allocation and release reload their slot from this base.
    rank, offset, base = (regs[key] for key in ("UR8", "UR9", "UR14"))
    definitions = {}
    for i, inst in enumerate(insts):
        if i in released:
            continue
        for reg in _registers(inst.get_defs()) & {rank, offset, base}:
            if reg in definitions or inst.pflag or i >= start:
                return
            definitions[reg] = i
    if set(definitions) != {rank, offset, base}:
        return
    start_rank, start_offset, build = (definitions[key] for key in (rank, offset, base))
    if (max(start_rank, start_offset) >= build or
            any(inst.is_branch() or inst.is_exit() for inst in insts[:build]) or
            any(any(dest != i + 1 and dest <= build for dest in nexts) for i, nexts in enumerate(edges)) or
            _tokens(_source(insts[start_rank])) != _tokens(f"S2UR {rank}, SR_CgaCtaId") or
            _tokens(_source(insts[build])) != _tokens(f"ULEA {base}, {rank}, {offset}, 0x18")):
        return
    load = insts[start_offset]
    if load.opcodes == ["UMOV"] and load.operands[1].is_immediate:
        return load.operands[1].immediate_value + regs["STATIC_DST"]


def _ranked_dead(insts, edges, join, matched, release):
    # Follow every consumer path. A register first used outside the protocol
    # must have been redefined after allocation on every incoming path.
    temps = set().union(*(_registers(insts[i].get_defs()) for i in matched))
    pending = [(join + 1, temps)]
    seen = {}
    while pending:
        i, live = pending.pop()
        live = live - seen.get(i, set())
        if not live:
            continue
        seen.setdefault(i, set()).update(live)
        inst = insts[i]
        if inst.is_call() or inst.is_return() or i in matched:
            return False
        uses = inst.get_uses() + ([inst.pflag] if inst.pflag else [])
        if live & _registers(uses):
            return False
        if i == release:
            continue  # its only input is the slot, and its successor exits
        if not inst.pflag:
            live -= _registers(inst.get_defs())
        if inst.opcodes[0] == "BSSY" and len(edges[i]) == 2:
            # BSSY installs a reconvergence PC and falls through. The edge to
            # that PC conservatively models a later divergence. Instructions
            # in the straight-line prefix execute before either path, e.g.
            # ptxas reloads the published slot into R2 immediately after BSSY.
            # Visit that prefix normally to check every read as well.
            target, successor = edges[i]
            reaching = live.copy()
            for j in range(successor, target):
                op = insts[j]
                if edges[j] != [j + 1] or op.is_branch() or op.is_call() or op.is_return() or op.is_exit():
                    break
                if not op.pflag:
                    reaching -= _registers(op.get_defs())
            pending.extend(((target, reaching), (successor, live)))
        else:
            pending.extend((n, live) for n in edges[i])
    return True


def _recover_ranked_pair(insts, by_pc):
    # Full CUDA 13 kernels may hoist the CTA-ranked base and use the uniform
    # parity test. This release validates the run and exits without issuing
    # a separate relinquish_alloc_permit.
    alloc_pat, free_pat = _ranked_patterns()
    sites = [i for i, inst in enumerate(insts) if inst.opcodes[0] == "UTCATOMSWS"]
    if len(sites) != 3:
        return
    start = sites[0] - next(i for i, (line, _) in enumerate(alloc_pat) if line.startswith("UTCATOMSWS"))
    free = sites[2] - next(i for i, (line, _) in enumerate(free_pat) if line.startswith("UTCATOMSWS"))
    alloc = _match(alloc_pat, insts, start, by_pc)
    release = _match(free_pat, insts, free, by_pc)
    if not alloc or not release:
        return
    allocated, _, regs = alloc
    released, keep, free_regs = release
    if (sites[1] not in allocated or allocated & released or
            any(regs[name] != free_regs[name] for name in ("MASK", "STATIC_DST", "UR8", "UR14"))):
        return
    operands = _alloc_operands(regs)
    edges = _ranked_edges(insts, by_pc)
    if not operands or edges is None:
        return
    # The original hot paths and both out-of-line wait/trap tails have one
    # permitted entry per collective. Keep the caller's first-warp guard.
    for i, nexts in enumerate(edges):
        for region, entry in ((allocated, start), (released, free)):
            if i not in region and any(n in region and n != entry for n in nexts):
                return
        if i not in released and (insts[i].is_call() or insts[i].is_return()):
            return
    offset = _ranked_base(insts, edges, start, released, regs)
    join = by_pc[regs["@join"]]
    if offset is None or offset % 4 or not 0 <= offset < 1 << 24 or not _ranked_dead(insts, edges, join, allocated, free):
        return
    slot = Operand.from_immediate(hex(offset), offset)
    columns = operands[1]
    address = insts[free].operands[0].clone()
    for i in (allocated | released) - keep - {free}:
        insts[i] = Instruction(insts[i].id, ["NOP"], [])
    insts[start] = Instruction(insts[start].id, ["TMEMALLOC2"], [slot, columns])
    insts[free + 1] = Instruction(insts[free + 1].id, ["TMEMFREE2", "DEALLOC"], [address, columns])
    return True


@lru_cache(maxsize=1)
def _ranked_guard_pattern():
    return _compile(_read_lines("129_ranked_guard"))


def _recover_guarded_ranked_pair(insts, by_pc, static_bytes):
    # The 12.9 full kernel performs a permit check before its elected
    # even/odd peer handoff and rebuilds its ranked base before release.
    from transform.tmem_column import _aligned_patterns, _guarded_ranked_slot

    sites = [i for i, inst in enumerate(insts) if inst.opcodes[0] == "UTCATOMSWS"]
    if len(sites) != 3:
        return
    alloc_pat, free_pat = _ranked_guard_pattern(), _aligned_patterns("129_multi")[1]
    def locate(pattern, site):
        # Labels bind PCs and consume no SASS instructions.
        prefix = 0
        for line, _ in pattern:
            if line.startswith("UTCATOMSWS"):
                break
            prefix += not line.endswith(":")
        index = site - prefix
        return index, _match(pattern, insts, index, by_pc) if index >= 0 else None
    start, alloc = locate(alloc_pat, sites[0])
    release, dealloc = locate(free_pat, sites[-1])
    if not alloc or not dealloc:
        return
    allocated, keep, regs = alloc
    released, _, free_regs = dealloc
    units = regs["UNITS"]
    if (sites[1] not in allocated or allocated & released or units not in (2, 4, 8, 16) or
            regs["MASK"] != (1 << units) - 1 or regs["MASK"] != free_regs["MASK"]):
        return
    offset = _guarded_ranked_slot(insts, by_pc, start, allocated, release, released, regs, free_regs,
                                  operand_names=(("UR11", "UR12"), ("UR13", "UR9")), alloc_base_source="UR10")
    if offset is None or offset < 0x400 or offset + 4 > static_bytes or offset % 4 or offset >= 1 << 24:
        return
    edges = _ranked_edges(insts, by_pc)
    join = by_pc[regs["@join"]]
    if edges is None or not _ranked_dead(insts, edges, join, allocated - keep, release):
        return
    slot = Operand.from_immediate(hex(offset), offset)
    columns = Operand.from_immediate(str(32 * units), 32 * units)
    # Preserve the CUDA permit check, sentinel, trap and returning join.
    # CuPBoP owns each peer's actual column allocation and validates the
    # published handle and size on release. Its handle is a paired address,
    # not a CUDA bitmap index.
    issue = by_pc[regs["@issue"]]
    exit_index = by_pc[free_regs["@exit"]] + 1
    for i in (allocated - keep) | (released - {release, exit_index}):
        insts[i] = Instruction(insts[i].id, ["NOP"], [])
    # The denied branch still stores -1 and executes the verified BPT.TRAP.
    # Inline that trap so call/return reconstruction never jumps into removed
    # deallocation tails that happen to surround the permit trap in SASS.
    trap = insts[by_pc[regs["@permit_trap"]]]
    call = next(i for i in keep if insts[i].is_call())
    insts[call] = Instruction(insts[call].id, trap.opcodes.copy(), [op.clone() for op in trap.operands])
    insts[issue] = Instruction(insts[issue].id, ["TMEMALLOC2"], [slot, columns])
    target = Operand.from_immediate(hex(regs["@join"]), regs["@join"])
    insts[issue + 1] = Instruction(insts[issue + 1].id, ["BRA"], [target])
    insts[release + 1] = Instruction(insts[release + 1].id, ["TMEMFREE2", "DEALLOC"],
                                     [insts[release].operands[0].clone(), columns])
    return True


def recover_tmem_pair(func, guarded_static_bytes=0):
    insts = func.instructions
    if not any(i.opcodes == ["UTCATOMSWS", "2CTA", "FIND_AND_SET", "ALIGN"] for i in insts):
        return
    by_pc = {int(inst.id, 16): i for i, inst in enumerate(insts)}
    if guarded_static_bytes and _recover_guarded_ranked_pair(insts, by_pc, guarded_static_bytes):
        return True
    if _recover_ranked_pair(insts, by_pc):
        return
    for start, first in enumerate(insts):
        if first.opcodes != ["S2R"] or "SR_CgaCtaId" not in _source(first):
            continue
        for pattern in _patterns():
            result = _match(pattern, insts, start, by_pc)
            if result is None:
                continue
            matched, keep, bindings = result
            operands = _alloc_operands(bindings)
            if operands is None:
                continue
            dst, columns = operands
            join = by_pc[bindings["@join"]]
            release = _paired_release(insts, join, by_pc, bindings, dst, columns)
            region = matched
            if release:
                free, address, free_sites, free_keep = release
                region = matched | set(range(join + 1, free)) | free_sites
            if not _single_entry(insts, start, region, by_pc) or not _scratch_dead(insts, matched, keep, join, release):
                continue
            for i in matched - keep:
                insts[i] = Instruction(insts[i].id, ["NOP"], [])
            insts[start] = Instruction(first.id, ["TMEMALLOC2"], [dst, columns])
            if release:
                for i in free_sites - free_keep:
                    insts[i] = Instruction(insts[i].id, ["NOP"], [])
                insts[free] = Instruction(insts[free].id, ["TMEMFREE2"], [address, columns])
            break
