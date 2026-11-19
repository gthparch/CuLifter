import re

from sir.controlcode import ControlCode
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import InvalidOperandException, Operand


class NoParsingEffort(Exception):
    pass


class UnmatchedControlCode(Exception):
    pass


class SaSSParserBase:
    def __init__(self, isa, file, elf):
        self.file = file
        self.elf = elf

    def apply(self):
        funcs = []
        curr_func = None
        insts = []

        lines = self.file.split("\n")
        modified_lines = []
        skip_next = False
        for i, line in enumerate(lines):
            if skip_next:
                skip_next = False
                continue
            if "{" in line:
                line = line.replace("{", "")
            if "}" in line:
                if line[-1] != "/":
                    line = line + lines[i + 1]
                    line = line.replace("\n", "")
                    skip_next = True
                line = line.replace("}", ";")
            modified_lines.append(line)

        lines = modified_lines

        for _, line in enumerate(lines):
            prev_func = curr_func
            curr_func = self.create_function(line, prev_func, insts)
            if prev_func != curr_func and curr_func is not None:
                if prev_func is not None:
                    prev_func.instructions = insts
                    funcs.append(prev_func)
                    insts = []
                continue
            curr_func = prev_func
            if curr_func is not None and ".headerflags" in line:
                curr_func.headerflags = self._parse_headerflags(line)
                continue
            self.parse_func_body(line, insts, curr_func)

        if curr_func is not None:
            curr_func.instructions = insts
            funcs.append(curr_func)

        return funcs

    def _parse_headerflags(self, line):
        match = re.search(r'@"([^"]+)"', line)
        return match.group(1)

    def create_function(self, line, prev_func, insts):
        if "/*" not in line or "*/" not in line:
            if "Function : " in line:
                items = line.split(" : ")
                return Function(items[1])
        return None

    def parse_func_body(self, line, insts, curr_func):
        raise NoParsingEffort

    def get_inst_num(self, line):
        items = line.split("/*")
        return items[1]

    def get_inst_opcode(self, line):
        content = line.split(";", 1)[0].lstrip()
        # nvdisasm attaches a BRX destination list to the operand. ResolveBRX
        # reads those destinations from the cubin's constant table.
        content = re.sub(r'\s*\(?\*"BRANCH_TARGETS [^"]*"\*\)?\s*$', '', content)
        opcode, *rest = content.split(maxsplit=1)
        operands = rest[0] if rest else ""
        # Match the whole guard token: P2R and P2R.B1 are instructions.
        guard = re.fullmatch(r"@?(!?)(U?P(?:[0-7]|T))", opcode)
        if guard is None:
            return opcode, None, operands
        prefix, register = guard.groups()
        opcode, *rest = operands.split(maxsplit=1)
        return opcode, Operand.from_reg(register, register, prefix=prefix or None), rest[0] if rest else ""

    def get_inst_operands(self, line):
        items = line.split(",")
        ops = []
        for item in items:
            operand = item.lstrip()
            if operand != "":
                ops.append(operand)
        return ops

    def parse_instruction(
        self,
        inst_id,
        opcode_content,
        pflag,
        operands_content,
        operands_detail,
        curr_func,
        encoding=None,
    ):
        opcodes = opcode_content.split(".")

        operands = []
        for operand_content in operands_content:
            operands.append(Operand.parse(operand_content))

        inst = Instruction(inst_id, opcodes, operands, None, pflag, encoding=encoding)
        raw_content = opcode_content
        if operands_detail:
            raw_content = f"{opcode_content} {operands_detail}"
        inst.inst_content = raw_content
        return inst

    def _extract_inst_encoding(self, line):
        matches = re.findall(r"/\*\s*(0x[0-9a-fA-F]+)\s*\*/", line)
        return matches[-1]

    def get_arg_offset(self, offset):
        offset = offset.replace("[", "")
        offset = offset.replace("]", "")
        return int(offset, base=16)

    def parse_control_code(self, content, control_codes):
        raise NoParsingEffort
