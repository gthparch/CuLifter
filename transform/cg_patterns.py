from transform.transform import SaSSTransform


class CGPatterns(SaSSTransform):
    TARGET_OPS = ("WARPSYNC", "VOTE", "VOTEU")

    def __init__(self):
        super().__init__()
        self.total_cg_patterns = 0
        self.cg_pattern_breakdown = {name: 0 for name in self.TARGET_OPS}

    def apply(self, module):
        super().apply(module)
        print("=== Start of CGPatterns ===")

        counts = {name: 0 for name in self.TARGET_OPS}
        for func in module.functions:
            for block in func.blocks:
                for inst in block.instructions:
                    if not inst.opcodes:
                        continue

                    opcode = inst.opcodes[0]
                    if opcode in counts:
                        counts[opcode] += 1

        self.cg_pattern_breakdown = counts
        self.total_cg_patterns = sum(counts.values())

        print("CG pattern counts:", self.cg_pattern_breakdown)
        
