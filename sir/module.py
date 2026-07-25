from sir.function import Function

class Module :
    def __init__(self, name, parser):
        self.name = name
        # Propagate ISA string from parser for arch-specific transforms
        self.isa = parser.isa
        # Provide ELF metadata to transform passes (e.g., relocation resolution).
        self.elf = parser.elf

        # Parse the module
        self.parse_module(parser)

    def parse_module(self, parser):
        self.functions = parser.apply()

    def __str__(self):
        lines = []
        if self.isa:
            lines.append(f"\tcode for {self.isa}")
        for func in self.functions:
            if lines:
                lines.append("")
            lines.append(str(func))
        return "\n".join(lines)

    def dump(self):
        lines = [""] if self.isa else []
        if self.isa:
            lines.append(f"\tcode for {self.isa}")
        for idx, func in enumerate(self.functions):
            if idx > 0:
                lines.append("")
            lines.extend(func.dump())
        if lines:
            lines.append("\t\t..........")
            lines.append("")
            lines.append("")
        return "\n".join(lines)
