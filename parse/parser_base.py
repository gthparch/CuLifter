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
        items = line.split(";")
        line = items[0].lstrip()
        items = line.split(" ")
        opcode = items[0]
        pflag = None

        pred_reg = None
        is_not = False
        if opcode.startswith("@"):
            opcode = opcode[1:]
        if opcode.startswith("!"):
            is_not = True
            opcode = opcode[1:]
        # P2R is an opcode, not a predicate guard — exclude it
        if opcode.startswith("UP") and len(opcode) > 2 and (opcode[2].isdigit() or opcode[2] == "T"):
            pred_reg = opcode
        elif opcode.startswith("P") and len(opcode) > 1 and (opcode[1].isdigit() or opcode[1] == "T"):
            if opcode not in ("P2R",):
                pred_reg = opcode

        rest_content = line.replace(items[0], "")

        if pred_reg:
            prefix = "!" if is_not else None
            pflag = Operand.from_reg(pred_reg, pred_reg, prefix=prefix)
            opcode = items[1]
            rest_content = rest_content.replace(items[1], "")

        return opcode, pflag, rest_content

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
