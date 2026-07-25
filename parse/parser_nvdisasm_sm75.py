import re

from sir.function import Function

from parse.parser_base import SaSSParserBase


_TYPE_RE = re.compile(r"^\s*\.type\s+(?P<name>[A-Za-z_][\w\.]*)\s*,\s*@function\b")
_LABEL_RE = re.compile(r"^\s*(?P<label>[$A-Za-z_\.][\w\.$]*)\s*:\s*$")
_INSTR_RE = re.compile(r"^\s*/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?P<body>.+?)\s*$")
_SECTION_RE = re.compile(r"^\s*\.section\s+(?P<name>[^\s,]+)")


class SaSSParser_NVDisasm_SM75(SaSSParserBase):
    def __init__(self, isa, file, elf):
        super().__init__(isa, file, elf)
        self._lines = file.splitlines()

    def apply(self):
        funcs = []
        pending_func = None
        curr_name = None
        curr_lines = []
        in_text_section = False

        for raw_line in self._lines:
            line = raw_line.strip()
            if not line:
                continue

            section_match = _SECTION_RE.match(line)
            if section_match:
                section_name = section_match.group("name")
                in_text_section = section_name.startswith(".text.")
                if not in_text_section and curr_name is not None and curr_lines:
                    label_map = {}
                    pending_labels = []
                    for func_line in curr_lines:
                        func_line = func_line.strip()
                        if not func_line:
                            continue
                        func_label = _LABEL_RE.match(func_line)
                        if func_label:
                            func_label_name = func_label.group("label")
                            if func_label_name.startswith(".L"):
                                pending_labels.append(func_label_name)
                            continue

                        instr_match = _INSTR_RE.match(func_line)
                        if not instr_match:
                            continue
                        inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                        if pending_labels:
                            for pending_label in pending_labels:
                                label_map[pending_label] = inst_id
                            pending_labels = []

                    curr_func = Function(curr_name)
                    insts = []
                    for func_line in curr_lines:
                        func_line = func_line.strip()
                        if not func_line:
                            continue
                        if _LABEL_RE.match(func_line):
                            continue
                        instr_match = _INSTR_RE.match(func_line)
                        if not instr_match:
                            continue

                        inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                        body = instr_match.group("body").strip()

                        body = body.replace("{", "").replace("}", "").strip()
                        if not body:
                            continue
                        if body.endswith(";"):
                            body = body[:-1].strip()

                        body = body.replace("`", "").replace("(", "").replace(")", "")
                        body = self._replace_label_refs(body, label_map)

                        opcode, pflag, rest = self.get_inst_opcode(body)
                        rest = rest.replace(" ", "")
                        rest = self._normalize_const_mem_operands(rest)
                        rest = self._normalize_mem_operands(rest)
                        operands = self.get_inst_operands(rest)
                        operands_detail = ",".join(operands)

                        inst = self.parse_instruction(
                            inst_id, opcode, pflag, operands, operands_detail, curr_func
                        )
                        insts.append(inst)

                    curr_func.instructions = insts
                    funcs.append(curr_func)
                    curr_name = None
                    curr_lines = []

            type_match = _TYPE_RE.match(line)
            if type_match:
                pending_func = type_match.group("name")
                continue

            label_match = _LABEL_RE.match(line)
            if label_match:
                label = label_match.group("label")
                if label.startswith(".L"):
                    if curr_name is not None:
                        curr_lines.append(raw_line)
                    continue
                if label.startswith("$"):
                    if curr_name is not None:
                        curr_lines.append(raw_line)
                    continue
                if label.startswith("."):
                    if pending_func is None:
                        continue
                if not in_text_section and pending_func is None:
                    continue
                if pending_func is None or label == pending_func:
                    if curr_name is not None:
                        label_map = {}
                        pending_labels = []
                        for func_line in curr_lines:
                            func_line = func_line.strip()
                            if not func_line:
                                continue
                            func_label = _LABEL_RE.match(func_line)
                            if func_label:
                                func_label_name = func_label.group("label")
                                if func_label_name.startswith(".L") or func_label_name.startswith("$"):
                                    pending_labels.append(func_label_name)
                                    continue

                            instr_match = _INSTR_RE.match(func_line)
                            if not instr_match:
                                continue
                            inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                            if pending_labels:
                                for pending_label in pending_labels:
                                    label_map[pending_label] = inst_id
                                pending_labels = []

                        curr_func = Function(curr_name)
                        insts = []
                        for func_line in curr_lines:
                            func_line = func_line.strip()
                            if not func_line:
                                continue
                            if _LABEL_RE.match(func_line):
                                continue
                            instr_match = _INSTR_RE.match(func_line)
                            if not instr_match:
                                continue

                            inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                            body = instr_match.group("body").strip()

                            body = body.replace("{", "").replace("}", "").strip()
                            if not body:
                                continue
                            if body.endswith(";"):
                                body = body[:-1].strip()

                            body = body.replace("`", "").replace("(", "").replace(")", "")
                            body = self._replace_label_refs(body, label_map)

                            opcode, pflag, rest = self.get_inst_opcode(body)
                            rest = rest.replace(" ", "")
                            rest = self._normalize_const_mem_operands(rest)
                            rest = self._normalize_mem_operands(rest)
                            operands = self.get_inst_operands(rest)
                            operands_detail = ",".join(operands)

                            inst = self.parse_instruction(
                                inst_id, opcode, pflag, operands, operands_detail, curr_func
                            )
                            insts.append(inst)

                        curr_func.instructions = insts
                        funcs.append(curr_func)

                    curr_name = label
                    curr_lines = []
                    pending_func = None
                continue

            if curr_name is None:
                continue
            if in_text_section:
                curr_lines.append(raw_line)

        if curr_name is not None:
            label_map = {}
            pending_labels = []
            for func_line in curr_lines:
                func_line = func_line.strip()
                if not func_line:
                    continue
                func_label = _LABEL_RE.match(func_line)
                if func_label:
                    func_label_name = func_label.group("label")
                    if func_label_name.startswith(".L") or func_label_name.startswith("$"):
                        pending_labels.append(func_label_name)
                    continue

                instr_match = _INSTR_RE.match(func_line)
                if not instr_match:
                    continue
                inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                if pending_labels:
                    for pending_label in pending_labels:
                        label_map[pending_label] = inst_id
                    pending_labels = []

            curr_func = Function(curr_name)
            insts = []
            for func_line in curr_lines:
                func_line = func_line.strip()
                if not func_line:
                    continue
                if _LABEL_RE.match(func_line):
                    continue
                instr_match = _INSTR_RE.match(func_line)
                if not instr_match:
                    continue

                inst_id = f"{int(instr_match.group('addr'), 16):04x}"
                body = instr_match.group("body").strip()

                body = body.replace("{", "").replace("}", "").strip()
                if not body:
                    continue
                if body.endswith(";"):
                    body = body[:-1].strip()

                body = body.replace("`", "").replace("(", "").replace(")", "")
                body = self._replace_label_refs(body, label_map)

                opcode, pflag, rest = self.get_inst_opcode(body)
                rest = rest.replace(" ", "")
                rest = self._normalize_const_mem_operands(rest)
                rest = self._normalize_mem_operands(rest)
                operands = self.get_inst_operands(rest)
                operands_detail = ",".join(operands)

                inst = self.parse_instruction(
                    inst_id, opcode, pflag, operands, operands_detail, curr_func
                )
                insts.append(inst)

            curr_func.instructions = insts
            funcs.append(curr_func)

        return funcs

    def _replace_label_refs(self, body, label_map):
        for label, addr in label_map.items():
            pattern = rf"(?<![\w.$]){re.escape(label)}(?![\w.$])"
            body = re.sub(pattern, f"0x{addr}", body)
        return body

    def _normalize_const_mem_operands(self, operand_str):
        # c[bank][Rn] -> c[bank][Rn+0x0]; immediate-index form needs no rewrite.
        operand_str = re.sub(
            r"c\[(0x[0-9a-fA-F]+)]\[(U?R\d+(?:\.[A-Za-z0-9]+)?)\]",
            r"c[\1][\2+0x0]",
            operand_str,
        )
        return operand_str

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
