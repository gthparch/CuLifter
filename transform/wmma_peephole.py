"""Recognise the 4-HMMA cluster that ptxas emits to implement one
m16n16k16 wmma.mma.sync, and replace those 4 instructions with a single
synthetic WMMA_MMA_M16N16K16 pseudo-op. The lifter has a matching
@lift_for handler that emits the high-level llvm.nvvm.wmma.m16n16k16.mma
intrinsic — letting llc / ptxas re-emit the correct SASS instead of us
trying to mirror it instruction-by-instruction.

Algorithm (def-use, NOT sequential — ptxas interleaves MOVM/LDG/HMMA):

  1. Collect every HMMA.1688 instruction.
  2. Build chains by walking C operand -> producer HMMA's D. A chain is
     one N-split (one column tile of the wmma D fragment).
  3. Pair chains that share A inputs at matching K-splits — each pair
     of length 2*N covers N consecutive wmma.mma calls (each pair of
     adjacent HMMAs along a chain is one K-split).
  4. For each (chain pair, K position) form a 4-HMMA group and replace
     it with a synthetic WMMA_MMA_M16N16K16 pseudo-op.

Only handles the row.row / row.col layouts (B-side MOVM presence on all
4 inputs or none). col.row / col.col would also need A-side MOVM
detection; those clusters are left untouched and lift via the per-HMMA
mma.m16n8k8 path.

Run AFTER SSA + RegRemap so register names are stable, but BEFORE
TypeAnalysis + lifting.
"""
from sir.instruction import Instruction
from transform.transform import SaSSTransform


_HMMA_OPCODE = "HMMA"
_HMMA_SHAPE = "1688"
_MOVM_OPCODE = "MOVM"
_MOVM_MODIFIER = "MT88"


class WmmaPeephole(SaSSTransform):
    def __init__(self, verbose=False):
        super().__init__(name="WmmaPeephole")
        self.verbose = verbose

    def apply(self, module):
        super().apply(module)
        for func in module.functions:
            if func.blocks:
                self._process_function(func)

    # ------------------------------------------------------------------
    # Pattern recognition
    # ------------------------------------------------------------------

    def _process_function(self, func):
        hmmas = []
        d_name_to_inst = {}
        for block in func.blocks:
            for inst in block.instructions:
                if (inst.opcodes
                        and inst.opcodes[0] == _HMMA_OPCODE
                        and len(inst.opcodes) > 1
                        and inst.opcodes[1] == _HMMA_SHAPE):
                    hmmas.append((block, inst))
                    d_base = self._reg_base(inst.get_defs()[0])
                    if d_base:
                        d_name_to_inst[d_base] = inst
        if not hmmas:
            return

        # Build chains via C[0]→producer-D edges. After SSA each D base name
        # is unique so the producer map is one-to-one.
        producer_by_consumer = {}
        chain_starts = []
        for _, inst in hmmas:
            uses = inst.get_uses()
            # HMMA.1688.F32 has uses [A0, A1, B0, C0..C3] (7 operands).
            if len(uses) < 7:
                continue
            c0_base = self._reg_base(uses[3])
            if not c0_base or c0_base.startswith("RZ"):
                chain_starts.append(inst)
                continue
            producer = d_name_to_inst.get(c0_base)
            if producer is None:
                # C0 doesn't trace to an HMMA D — likely a phi /
                # loop-carried accumulator — treat as chain start.
                chain_starts.append(inst)
            else:
                producer_by_consumer[inst] = producer

        consumer_by_producer = {}
        for cons, prod in producer_by_consumer.items():
            consumer_by_producer.setdefault(prod, []).append(cons)

        chains = []
        for start in chain_starts:
            chain = [start]
            cur = start
            while cur in consumer_by_producer:
                nxt = consumer_by_producer[cur]
                if len(nxt) != 1:
                    break  # fan-out; punt.
                cur = nxt[0]
                chain.append(cur)
            chains.append(chain)

        # Pair chains by identical A-signature (the same A operand pair at
        # each matching K-split position) — these are the two N-splits of
        # the same wmma stream.
        sigs = [tuple(self._a_signature(c) for c in chain) for chain in chains]
        pairs = []
        used = set()
        for i in range(len(chains)):
            if i in used:
                continue
            for j in range(i + 1, len(chains)):
                if j in used:
                    continue
                if sigs[i] == sigs[j] and len(chains[i]) == len(chains[j]):
                    pairs.append((i, j))
                    used.add(i)
                    used.add(j)
                    break

        # Identify MOVM destinations for the layout check below.
        movm_dests = set()
        for block in func.blocks:
            for inst in block.instructions:
                if (inst.opcodes and inst.opcodes[0] == _MOVM_OPCODE
                        and _MOVM_MODIFIER in inst.opcodes):
                    d_base = self._reg_base(inst.get_defs()[0])
                    if d_base:
                        movm_dests.add(d_base)

        # Form 4-HMMA groups and replace each with a synthetic pseudo-op.
        groups = []
        for i, j in pairs:
            ci, cj = chains[i], chains[j]
            for k in range(len(ci) // 2):
                groups.append((ci[2*k], ci[2*k + 1], cj[2*k], cj[2*k + 1]))

        replaced = 0
        for group in groups:
            if self._group_layout_supported(group, movm_dests):
                self._splice_group(group)
                replaced += 1

        if self.verbose:
            print(f"[WmmaPeephole] {func.name[:50]}: "
                  f"{len(hmmas)} HMMA.1688, {len(chains)} chain(s), "
                  f"{len(pairs)} pair(s), {replaced}/{len(groups)} replaced")

    # ------------------------------------------------------------------
    # Layout check + rewrite
    # ------------------------------------------------------------------

    @staticmethod
    def _group_layout_supported(group, movm_dests):
        """row.row (all 4 B operands MOVM-fed) and row.col (none) are both
        handled by the same synthetic pseudo-op — the lifter emits
        wmma.row.col since the operand we read is post-MOVM (col-major)
        in either case. Mixed / col.* clusters need A-MOVM handling that
        isn't implemented; leave them as separate HMMAs."""
        n_movm = sum(1 for h in group
                     if WmmaPeephole._reg_base(h.get_uses()[2]) in movm_dests)
        return n_movm in (0, 4)

    def _splice_group(self, group):
        """Build a WMMA_MMA_M16N16K16 pseudo-op carrying the group's 16 D
        defs and 16 inputs (4 A + 4 B + 8 C), insert it at the position
        of the LAST HMMA in block order, and remove all 4 HMMAs."""
        block = group[-1].parent

        # Defs: 4 chain × 4 D regs each. We carry all 16 even though only
        # group[1] / group[3] hold the chain-final D values — the
        # intermediate D defs (group[0], group[2]) are SSA-unique names
        # whose later uses (if any) need to point to the same value the
        # wmma fragment produces, so the lift writes the same float into
        # both. After RegRemap + DCE the dead intermediate writes
        # disappear.
        defs = []
        for hmma in group:
            for d in hmma.get_defs():
                defs.append(d.clone())

        # Inputs: per-thread fragment layout for sm_75 wmma m16n16k16.
        # A: 4 unique v2half (2 from each of group[0] / group[1]'s K-split).
        # B: 4 unique v2half (one per group, post-MOVM = col-major).
        # C: 8 f32 — 4 from chain A's initial accumulator (group[0].C),
        #    4 from chain B's initial accumulator (group[2].C).
        a_ops = list(group[0].get_uses()[:2]) + list(group[1].get_uses()[:2])
        b_ops = [g.get_uses()[2] for g in group]
        c_ops = list(group[0].get_uses()[3:7]) + list(group[2].get_uses()[3:7])
        uses = [op.clone() for op in (a_ops + b_ops + c_ops)]

        # Synthetic opcode follows SASS modifier convention: base opcode +
        # variant tokens in opcodes[1:]. We only emit one variant today
        # (the cluster pattern recognised here is m16n16k16 f16→f32 row.row
        # / row.col, both routed to row.col on the lift side because B is
        # post-MOVM = col-major fragment storage). Other shapes
        # (m32n8k16, m8n32k16), accumulator types (f16), or A/B layouts
        # (col.row, col.col) will be added by extending the recogniser
        # and emitting different modifier tokens; the lift dispatches on
        # those tokens.
        anchor_id = group[-1].id
        new_inst = Instruction(
            id=f"{anchor_id}_wmma",
            opcodes=["WMMA_MMA", "M16N16K16", "ROW", "COL", "F32", "F32"],
            operands=defs + uses,
            parentBB=block,
        )
        new_inst.use_op_start_idx = len(defs)

        last_idx = max(block.instructions.index(h) for h in group)
        block.instructions[last_idx] = new_inst
        for h in group:
            if h is not group[-1] and h in block.instructions:
                block.instructions.remove(h)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _a_signature(inst):
        uses = inst.get_uses()
        if len(uses) < 2:
            return None
        return (WmmaPeephole._reg_base(uses[0]),
                WmmaPeephole._reg_base(uses[1]))

    @staticmethod
    def _reg_base(operand):
        if operand is None:
            return None
        if getattr(operand, "is_reg_group", False):
            sub = operand.reg_group[0] if operand.reg_group else None
            return sub.reg if sub else None
        return getattr(operand, "reg", None)
