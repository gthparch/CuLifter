import random

from transform.transform import SaSSTransform


class ToggleReuse(SaSSTransform):
    def __init__(self, probability=1 / 2):
        super().__init__()
        self.probability = probability

    def apply(self, module):
        for func in module.functions:
            if func.blocks:
                for block in func.blocks:
                    self._process_instructions(block.instructions)
            else:
                self._process_instructions(func.instructions)

    def _process_instructions(self, instructions):
        for inst in instructions:
            is_mem_inst = inst.opcodes[0].startswith("LD") or inst.opcodes[0].startswith("ST")
            for operand in inst.get_uses():
                if not operand.is_writable_reg:
                    continue
                if operand.is_predicate_reg:
                    continue
                if operand.is_mem_addr:
                    continue
                if is_mem_inst:
                     continue
                if random.random() < self.probability:
                    operand.reuse = not operand.reuse
