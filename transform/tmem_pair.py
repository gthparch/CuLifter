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
_TOKEN = re.compile(r"@[a-z_]+|\w+|[^\w\s]")


def _tokens(line):
    return _TOKEN.findall(line.replace(".reuse", ""))


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
    return [_compile(lines) for lines in patterns]


@lru_cache(maxsize=2)
def _release_pattern(single):
    return _compile(_read_lines("free_single" if single else "free_multi"))


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
        elif (name.startswith("@") and name != "@") or name in ("DST", "STATIC_DST", "NCOLS", "UNITS", "MASK"):
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
            label = "@" + line[:-1]
            if label == "@odd_elect":
                # The odd CTA reloads its own destination after reconvergence.
                # The even CTA's uniform allocation temporaries are now dead.
                for reg in list(bindings):
                    if reg.startswith("UR"):
                        reverse.pop(bindings.pop(reg), None)
            if label in ("@even_slow", "@odd_slow", "@trap_start", "@trap_mask"):
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


def _single_entry(instructions, start, matched, by_pc):
    # A branch into either retry/hand-off path would bypass the collective.
    for i, inst in enumerate(instructions):
        if i in matched:
            continue
        if inst.opcodes[0] in ("BRX", "JMX", "JMXU", "BSSY", "SSY", "PRET"):
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
    writers = {}
    for i in range(join + 1, len(insts)):
        inst = insts[i]
        if inst.opcodes == ["LDS"] and inst.get_uses() and inst.get_uses()[0].offset == 0x14:
            break
        if inst.pflag or inst.opcodes[0] not in _CONSUMER_OPS - {"STS"}:
            return
        for reg in _registers(inst.get_defs()):
            writers[reg] = i
    else:
        return

    ncols = columns.immediate_value
    result = _match(_release_pattern(ncols == 32), insts, i, by_pc)
    if result is None:
        return
    sites, keep, regs = result
    if regs["R0"] != bindings["R0"] or regs["R0"] in writers:
        return
    mask_index, address_index = writers.get(regs["R9"]), writers.get(regs["R5"])
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


def recover_tmem_pair(func):
    insts = func.instructions
    if not any(i.opcodes == ["UTCATOMSWS", "2CTA", "FIND_AND_SET", "ALIGN"] for i in insts):
        return
    by_pc = {int(inst.id, 16): i for i, inst in enumerate(insts)}
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
