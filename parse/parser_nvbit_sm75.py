import re

from sir.basicblock import BasicBlock
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import Operand

from parse.parser_base import SaSSParserBase


_INSTR_LINE_RE = re.compile(
    r"^Instr\s+(?P<idx>\d+)\s+@\s+(?P<addr>0x[0-9a-fA-F]+)\s+\(\d+\)\s+-\s+(?P<body>.+?)\s*;?\s*$"
)


class SaSSParser_NVBit_SM75(SaSSParserBase):
    def __init__(self, isa, file, elf):
        super().__init__(isa, file, elf)
        self._lines = file.splitlines()

    def apply(self):
        funcs = []
        curr_func = None
        insts = []

        def finalize_current():
            nonlocal curr_func, insts
            if curr_func and insts:
                while insts and insts[-1].is_nop():
                    insts.pop()
                if not insts:
                    curr_func = None
                    insts = []
                    return
                last_inst = insts[-1]
                if not (last_inst.is_exit() or last_inst.is_return() or last_inst.is_branch()):
                    new_id = f"{int(last_inst.id, 16) + 1:04x}"
                    insts.append(Instruction(new_id, ["EXIT"], [], None, None))
                curr_func.blocks = self._create_cfg(insts)
                funcs.append(curr_func)
            curr_func = None
            insts = []

        for raw_line in self._lines:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("inspecting "):
                finalize_current()
                name = self._parse_function_name(line, len("inspecting "))
                curr_func = Function(name)
                insts = []
                continue

            if line.startswith("Function "):
                finalize_current()
                name = self._parse_function_name(line, len("Function "))
                curr_func = Function(name)
                insts = []
                continue

            if curr_func is None:
                continue

            if line.startswith("kernel "):
                finalize_current()
                continue

            if line.startswith("Basic block id") or line.startswith("Basic Block ID"):
                continue

            if line.startswith("Inject ") or line.startswith("Load module"):
                continue

            match = _INSTR_LINE_RE.match(line)
            if not match:
                continue

            inst_id = f"{int(match.group('addr'), 16):04x}"
            body = match.group("body").rstrip()
            if body.endswith(";"):
                body = body[:-1].rstrip()

            opcode, pflag, rest = self.get_inst_opcode(body)
            rest = rest.replace(" ", "")
            rest = self._normalize_const_mem_operands(rest)
            rest = self._normalize_mem_operands(rest)
            operands = self.get_inst_operands(rest)
            operands = self._strip_branch_predicate(opcode, operands)
            operands_detail = ",".join(operands)
            inst = self.parse_instruction(inst_id, opcode, pflag, operands, operands_detail, curr_func)

            insts.append(inst)

        finalize_current()
        return funcs

    def _parse_function_name(self, line, prefix_len):
        content = line[prefix_len:].strip()
        if " - " in content:
            content = content.split(" - ", 1)[0].strip()
        return content

    def _strip_branch_predicate(self, opcode, operands):
        if opcode not in ("BRA", "PBRA"):
            return operands

        idx = 0
        while idx < len(operands) and self._is_predicate_operand(operands[idx]):
            idx += 1
        return operands[idx:]

    def _is_predicate_operand(self, operand):
        if operand.startswith("P") and operand[1:].isdigit():
            return True
        if operand.startswith("!P") and operand[2:].isdigit():
            return True
        if operand in ("PT", "!PT"):
            return True
        return False

    def _normalize_const_mem_operands(self, operand_str):
        # c[bank][Rn] -> c[bank][Rn+0x0]; immediate-index form needs no rewrite.
        operand_str = re.sub(
            r"c\[(0x[0-9a-fA-F]+)]\[(U?R\d+(?:\.[A-Za-z0-9]+)?)\]",
            r"c[\1][\2+0x0]",
            operand_str,
        )
        return operand_str

    def _create_cfg(self, insts):
        addr_set  = {int(inst.id, 16) for inst in insts}
        addr_list = sorted(addr_set)

        def _align_up_to_inst(addr_hex):
            addr = int(addr_hex, 16)
            max_addr = addr_list[-1]

            while addr not in addr_set and addr <= max_addr:
                addr += 0x8
            return addr

        for inst in insts:
            if inst.is_branch():
                target_addr = _align_up_to_inst(inst.operands[-1].name.zfill(4))
                inst.operands[0].name = format(target_addr, '04x')
                inst.operands[0].immediate_value = target_addr

        leaders = set()
        predicated_leaders = set()
        curr_pred = None
        for i, inst in enumerate(insts):
            if inst.is_return() or inst.is_exit():
                if i + 1 < len(insts):
                    leaders.add(insts[i + 1].id)
            if inst.is_branch():
                leaders.add(inst.operands[0].name.zfill(4))
            if curr_pred != inst.pflag:
                leaders.add(inst.id)
                if inst.predicated():
                    predicated_leaders.add(inst.id)
                curr_pred = inst.pflag

        blocks = []
        block_insts = []
        pflag = None
        prev_block = None
        next_block = {}
        block_by_addr = {}
        predicated_blocks = set()

        for inst in insts:
            if inst.id in leaders:
                if block_insts:
                    block_id = f"{int(block_insts[0].id, 16):04X}"
                    predicated_block = False
                    if block_insts[0].id in predicated_leaders:
                        predicated_block = True
                        pbra_inst = Instruction(
                            id=f"{int(block_insts[0].id, 16):04X}",
                            opcodes=["PBRA"],
                            operands=[block_insts[0].pflag.clone()],
                            parentBB=None,
                            pflag=None
                        )
                        pbra_inst.inst_content = f"PBRA {block_insts[0].pflag.name}"
                        block_insts.insert(0, pbra_inst)
                        block_insts[1].id = f"{int(block_insts[0].id, 16)+1:04X}"
                        for pred_inst in block_insts:
                            pred_inst.pflag = None

                    block = BasicBlock(block_id, pflag, block_insts)
                    blocks.append(block)

                    if prev_block:
                        next_block[prev_block] = block
                    prev_block = block

                    block_by_addr[block.addr_content] = block

                    if predicated_block:
                        predicated_blocks.add(block)

                block_insts = [inst]
            else:
                block_insts.append(inst)

        if block_insts:
            block_id = f"{int(block_insts[0].id, 16):04X}"
            block = BasicBlock(block_id, pflag, block_insts)
            blocks.append(block)
            if prev_block:
                next_block[prev_block] = block
            block_by_addr[block.addr_content] = block

        for idx in range(len(blocks) - 1):
            if blocks[idx] not in next_block:
                next_block[blocks[idx]] = blocks[idx + 1]

        for block in blocks:
            terminator = block.get_terminator()

            if not terminator:
                successor = next_block.get(block)
                if not successor:
                    continue
                dest_op = Operand.from_immediate(successor.addr_content, int(successor.addr_content, 16))
                new_inst = Instruction(
                    id=f"{int(block.instructions[-1].id, 16)+1:04X}",
                    opcodes=["BRA"],
                    operands=[dest_op]
                )
                block.instructions.append(new_inst)

                block.add_succ(successor)
                successor.add_pred(block)
            elif terminator.is_branch():
                dest_op = terminator.get_uses()[0]
                successor = block_by_addr[dest_op.name]
                block.add_succ(successor)
                successor.add_pred(block)
            elif terminator.is_return() or terminator.is_exit():
                pass
            else:
                raise Exception("Unrecognized terminator")

        insert_block = {}
        for block in predicated_blocks:
            pbra_inst = block.instructions[0]
            new_block_insts = block.instructions[1:]
            new_block = BasicBlock(new_block_insts[0].id, None, new_block_insts)
            block.instructions = [pbra_inst]

            insert_block[block] = new_block
            block_by_addr[new_block.addr_content] = new_block

            new_block.succs = block.succs
            block.succs = []
            for succ in new_block.succs:
                succ.preds.remove(block)
                succ.add_pred(new_block)

            new_block.add_pred(block)
            block.add_succ(new_block)

            false_block = next_block.get(block)
            if false_block:
                block.add_succ(false_block)
                false_block.add_pred(block)
                pbra_inst.operands.append(Operand.from_immediate(new_block.addr_content, int(new_block.addr_content, 16)))
                pbra_inst.operands.append(Operand.from_immediate(false_block.addr_content, int(false_block.addr_content, 16)))

        final_blocks = []
        for block in blocks:
            final_blocks.append(block)
            if block in insert_block:
                final_blocks.append(insert_block[block])

        for block in final_blocks:
            for inst in block.instructions:
                inst.parent = block

        return final_blocks

    def _normalize_mem_operands(self, operand_str):
        operand_str = re.sub(
            r"\[(R[0-9]+(?:\.[A-Z0-9]+)?)\+UR(\d+)\+(0x[0-9a-fA-F]+)\]",
            r"[\1+\3+UR\2]",
            operand_str,
        )
        operand_str = re.sub(
            r"\[(R[0-9]+(?:\.[A-Z0-9]+)?)\+UR(\d+)]",
            r"[\1+0x0+UR\2]",
            operand_str,
        )
        return operand_str
