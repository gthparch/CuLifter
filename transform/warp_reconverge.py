"""Insert warp-reconvergence at the merge of predicated shared-memory regions.

Root cause (cg_warpreduce 32->13, graphExecution, pns mean_maxs=0):
ptxas leaves warp-synchronous reductions' intra-warp steps BARRIER-FREE and
relies on the warp executing in lockstep — the predicated shared store/load
(`@P STS`/`@P LDS`) is a PREDICATED INSTRUCTION, so all 32 lanes stay converged
and step k+1's cross-lane `ld.shared` sees step k's `st.shared`. BuildCFG lowers
the predicate by BRANCH-SPLITTING into a PBRA-guarded block (`@%p bra`); on
Volta+ (independent thread scheduling) the warp then DIVERGES and does not
reconverge until an explicit sync, so the reduction reads stale lanes.

Fix: at the merge of a predicated region whose body contains a shared-memory
op, insert a synthetic WARPRECONV (lifted to `bar.warp.sync`). This restores the
warp-synchronous semantics the original SASS relied on. It is WARP-level (not a
CTA barrier), so it is deadlock-safe: every lane that entered the predicated
region reconverges at the merge — there is no cross-warp wait that could hang in
a thread-divergent context (which is what makes a blanket CTA `bar.sync` unsafe).

Scoped to SHARED-memory ops only (STS/STSM/LDS/LDSM) — the warp-synchronous
reduction idiom is shared-memory; global ops have their own ordering via fences.
"""
from transform.transform import SaSSTransform
from sir.instruction import Instruction

_SHARED_MEM_OPS = {"STS", "STSM", "LDS", "LDSM"}


class WarpReconverge(SaSSTransform):
    def apply(self, module):
        total = 0
        for func in module.functions:
            total += self._process(func)
        if total:
            print(f"WarpReconverge: inserted {total} warp-sync reconvergence point(s)")

    def _process(self, func):
        by_addr = {}
        for b in func.blocks:
            by_addr[b.addr_content] = b
            by_addr[b.addr_content.lower()] = b

        count = 0
        for block in func.blocks:
            if not block.instructions:
                continue
            pbra = block.instructions[0]
            if not (pbra.opcodes and pbra.opcodes[0] == "PBRA"):
                continue
            if len(pbra.operands) < 3:
                continue
            # PBRA operands = [pflag, true_br(body), false_br(merge)]
            body = by_addr.get(pbra.operands[1].name) or by_addr.get(pbra.operands[1].name.lower())
            merge = by_addr.get(pbra.operands[2].name) or by_addr.get(pbra.operands[2].name.lower())
            if body is None or merge is None:
                continue
            if not any(i.opcodes and i.opcodes[0] in _SHARED_MEM_OPS
                       for i in body.instructions):
                continue
            # Idempotent: don't double-insert if merge already starts with one.
            first = merge.instructions[0] if merge.instructions else None
            if first and first.opcodes and first.opcodes[0] == "WARPRECONV":
                continue
            ws = Instruction(
                id=f"{merge.addr_content}_wreconv",
                opcodes=["WARPRECONV"],
                operands=[],
            )
            ws.parent = merge
            merge.instructions.insert(0, ws)
            count += 1
        return count
