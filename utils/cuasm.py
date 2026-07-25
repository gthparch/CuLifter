import re


class CuAsm:
    def __init__(self, text):
        self.text = text
        self.lines = text.splitlines()
        self.section_re = re.compile(r"^\s*\.section\s+(?P<name>\.[^\s,]+)")
        self.addr_re = re.compile(r"/\*(?P<addr>[0-9a-fA-F]+)\*/\s*(?P<body>.+)$")
        self.text_sections = {}

    def set_nvdisasm(self, text):
        self.text_sections = {}
        curr_section = None
        in_text = False
        for line in text.splitlines():
            match = self.section_re.match(line)
            if match:
                curr_section = match.group("name")
                in_text = curr_section.startswith(".text.")
                continue
            if not in_text:
                continue
            match = self.addr_re.search(line)
            if not match:
                continue
            addr = int(match.group("addr"), 16)
            body = match.group("body").strip()
            section_map = self.text_sections.setdefault(curr_section, {})
            section_map[addr] = body

    def set_module(self, module):
        self.text_sections = {}
        for func in module.functions:
            insts = []
            if func.blocks:
                for block in func.blocks:
                    insts.extend(block.instructions)
            else:
                insts.extend(func.instructions)
            section_name = f".text.{func.name}"
            section_map = self.text_sections.setdefault(section_name, {})
            for inst in insts:
                line = inst.dump()[0]
                match = self.addr_re.search(line)
                addr = int(match.group("addr"), 16)
                body = match.group("body").strip()
                section_map[addr] = body

    def emit(self):
        out_lines = []
        curr_section = None
        in_text = False
        for line in self.lines:
            match = self.section_re.match(line)
            if match:
                curr_section = match.group("name")
                in_text = curr_section.startswith(".text.")
                out_lines.append(line)
                continue
            if in_text and curr_section in self.text_sections:
                match = self.addr_re.search(line)
                if match:
                    addr = int(match.group("addr"), 16)
                    body = self.text_sections[curr_section].get(addr)
                    if body:
                        end = line.find("*/", match.start())
                        prefix = line[: end + 2]
                        out_lines.append(f"{prefix} {body}")
                        continue
            out_lines.append(line)
        return "\n".join(out_lines) + "\n"
