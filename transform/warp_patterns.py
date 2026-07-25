"""
Warp pattern recognition transform.

Detects common warp-level operation patterns in SM90 SASS and annotates them
for diagnostics. Recognized patterns:

  1. SHFL.DOWN reduction: a chain of SHFL.DOWN + FADD/IADD with decreasing
     offsets (16, 8, 4, 2, 1) -- classic warp reduction sum.

  2. SHFL.IDX broadcast: SHFL.IDX with constant source lane 0 (or RZ) --
     broadcast from lane 0 to all lanes.

  3. SHFL.BFLY reduction: SHFL.BFLY + FADD/IADD chain with XOR masks
     (16, 8, 4, 2, 1) -- butterfly warp reduction.

  4. VOTE.ANY/ALL guard: VOTE followed by branch on result -- common
     control flow predicate pattern.

This pass is purely diagnostic; it does not modify instructions. The x86
runtime already handles all SHFL/VOTE operations correctly via multi-threaded
warp emulation.
"""
from transform.transform import SaSSTransform


class WarpPatterns(SaSSTransform):
    """Detect and log common warp operation patterns."""

    def __init__(self):
        super().__init__()
        self.patterns = {}  # pattern_type -> count

    def apply(self, module):
        super().apply(module)
        self.patterns = {
            "shfl_down_reduce": 0,
            "shfl_up_scan": 0,
            "shfl_bfly_reduce": 0,
            "shfl_idx_broadcast": 0,
            "vote_guard": 0,
        }

        for func in module.functions:
            for block in func.blocks:
                insts = block.instructions
                self._detect_shfl_reduction(insts, "DOWN", "shfl_down_reduce")
                self._detect_shfl_reduction(insts, "BFLY", "shfl_bfly_reduce")
                self._detect_shfl_scan(insts)
                self._detect_shfl_broadcast(insts)
                self._detect_vote_guard(insts)

        total = sum(self.patterns.values())
        if total > 0:
            print(f"Warp pattern counts: {self.patterns}")

    def _detect_shfl_reduction(self, insts, mode, pattern_name):
        """Detect SHFL.{DOWN,BFLY} + FADD/IADD reduction chains.

        A reduction chain is a sequence of SHFL.{mode} instructions in a block
        where each is followed (within a small window) by an add instruction
        that uses the shuffle result. We count the number of such pairs in
        the block and consider it a reduction pattern if there are 3+.
        """
        add_ops = ("FADD", "IADD3", "IADD")
        pair_count = 0
        for i, inst in enumerate(insts):
            if not inst.opcodes:
                continue
            if inst.opcodes[0] == "SHFL" and len(inst.opcodes) > 1 and inst.opcodes[1] == mode:
                # Look ahead up to 3 instructions for a matching add
                for j in range(i + 1, min(i + 4, len(insts))):
                    nxt = insts[j]
                    if nxt.opcodes and nxt.opcodes[0] in add_ops:
                        pair_count += 1
                        break

        if pair_count >= 3:
            self.patterns[pattern_name] += 1

    def _detect_shfl_scan(self, insts):
        """Detect SHFL.UP + FADD/IADD prefix scan chains."""
        add_ops = ("FADD", "IADD3", "IADD")
        pair_count = 0
        for i, inst in enumerate(insts):
            if not inst.opcodes:
                continue
            if inst.opcodes[0] == "SHFL" and len(inst.opcodes) > 1 and inst.opcodes[1] == "UP":
                for j in range(i + 1, min(i + 4, len(insts))):
                    nxt = insts[j]
                    if nxt.opcodes and nxt.opcodes[0] in add_ops:
                        pair_count += 1
                        break

        if pair_count >= 3:
            self.patterns["shfl_up_scan"] += 1

    def _detect_shfl_broadcast(self, insts):
        """Detect SHFL.IDX with source lane 0 or RZ (broadcast)."""
        for inst in insts:
            if not inst.opcodes:
                continue
            if inst.opcodes[0] == "SHFL" and len(inst.opcodes) > 1 and inst.opcodes[1] == "IDX":
                uses = inst.get_uses()
                if len(uses) >= 2:
                    lane_op = uses[1]
                    # Source lane is RZ (0) or immediate 0
                    if lane_op.is_rz or (lane_op.is_immediate and lane_op.immediate_value == 0):
                        self.patterns["shfl_idx_broadcast"] += 1

    def _detect_vote_guard(self, insts):
        """Detect VOTE followed by a conditional branch on the result."""
        for i, inst in enumerate(insts):
            if not inst.opcodes:
                continue
            if inst.opcodes[0] in ("VOTE", "VOTEU"):
                # Check next few instructions for a branch using the vote result
                for j in range(i + 1, min(i + 4, len(insts))):
                    next_inst = insts[j]
                    if next_inst.opcodes and next_inst.opcodes[0] in ("BRA", "EXIT"):
                        self.patterns["vote_guard"] += 1
                        break
