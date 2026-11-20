from collections import deque
from transform.transform import SaSSTransform
from sir.operand import Operand
from sir.instruction import Instruction

class DCE(SaSSTransform):
    def __init__(self, extra_side_effect=None, name=None):
        super().__init__(name=name)
        # Backend-scoped extra opcodes to treat as side-effecting (kept live).
        # Default empty -> DCE behaves exactly as before for every existing
        # caller. NVVMLifter uses this for SM90 cluster barriers and warpgroup
        # MMA instructions, which can change synchronization state even when
        # their register results are unused.
        self._extra_side_effect = frozenset(extra_side_effect or ())

    def apply(self, module):
        print("=== Start of DCE ===")

        for func in module.functions:
            print(f"Processing function: {func.name}")
            self.process(func)

        print("=== End of DCE ===")

    def process(self, func):
        side_effect_instructions = set([
            "STG",
            "ST",
            "STS",
            "STSM",
            "SUST",
            "STL",
            "STP",
            "LDGSTS",
            # TMA bulk copy global→shared (cp.async.bulk): writes shared
            # memory, defines no registers — the LDGSTS situation exactly.
            # Without this, FA3-style kernels lose their smem tile fill and
            # convert zeros (fa_oracle ConvertdQ finding, 2026-06-12).
            "UBLKCP",
            # TMA tensor bulk copy global→shared (cp.async.bulk.tensor /
            # UTMALDG): same as UBLKCP — writes shared memory, defines no
            # register, so DCE would delete it and the HGMMA operands would
            # never be staged into smem. This is THE FlashAttention blocker.
            "UTMALDG",
            "UTMASTG",
            # TMA bulk REDUCE-add global += shared, fp32 (UBLKRED): writes global
            # memory, defines no register — same situation as UTMASTG. Without
            # this, FA3 BACKWARD loses its dQ_accum accumulation (the only writer
            # of dQ_accum), so postprocess converts zeros and dQ = all zeros.
            "UBLKRED",

            "MEMBAR",
            "DEPBAR",  
            
            "BAR",
            "SYNC",
            "WARPSYNC",
            # Synthetic warp-reconvergence (WarpReconverge pass): a bar.warp.sync
            # inserted at the merge of a predicated shared-memory region so the
            # warp reconverges after a branch-split predicated store/load.
            # Defines no register; must not be DCE'd or the reconvergence is lost.
            "WARPRECONV",
            # Warp-collective ops: every lane must execute them for the warp to
            # converge, so they are side-effecting even when an individual lane's
            # result looks dead. Dropping them desynchronizes the warp and (for
            # SHFL) silently deletes scan steps — e.g. nvtiff LZW lost ~half its
            # SHFL.UP prefix-scan shuffles, corrupting output positions.
            "SHFL",
            "LDTM",
            "UTCIMMA",
            "UTCATOMSWS",
            "TMEMALLOC2",
            "TMEMFREE2",
            "UTCBAR",
            "UGETNEXTWORKID",
            "VOTE",
            "VOTEU",
            "ATOMS",
            "ATOMG",
            "ATOM",
            
            "BPT",
            "ATOM",
            "RED",
            "REDG",
            "CALL",
            "CALLFUNC",
            "CALLEXT",
            "RET",
            "EXIT",
            "BRA",
            "SWITCH",
            "PBRA",

            "SYNCS",
            "STAS",
            "REDAS",
            "NANOSLEEP",
            "FENCE",
            # ARRIVES.LDGSTSBAR.TRANSCNT — mbarrier arrive-on-cp.async. A sync
            # marker like SYNCS; defines no register, so without this DCE could
            # treat it as dead. Lifts to a NOP on x86 (lift_arrives), but keep it
            # as a live anchor for consistency with the other mbarrier/async ops.
            "ARRIVES",
            # SM90 warpgroup-MMA bracket ops: WARPGROUP.ARRIVE -> wgmma.fence,
            # WARPGROUP.DEPBAR.LE -> scoreboard wait. No
            # defs, no reg uses — DCE silently deleted every one, leaving the
            # lifted PTX with ZERO wgmma fences/waits: the async MMA unit read
            # A/accumulator registers unfenced (FA3 P@V produced deterministic
            # partially-stale O; small kernels survived only because ptxas
            # injects best-effort warpgroup.arrive/wait itself). Same for
            # ACQBULK (-> cp.async.bulk.wait_group 0: the TMA-store drain
            # before smem reuse) and UTMACMDFLUSH (TMA command flush).
            "WARPGROUP",
            "ACQBULK",
            "UTMACMDFLUSH",
        ])
        # Backend-scoped additions (empty for x86 -> no behavior change).
        side_effect_instructions |= self._extra_side_effect

        # `live` used to be a list; `def_inst not in live` is O(N) and made
        # DCE quadratic in instruction count. For functions with thousands
        # of instructions (especially after HandleCall inlining) this was
        # the dominant cost of lifting — observed as a >30-min hang on
        # hecbench/frechet-cuda. Use a set for membership, keep list
        # only for consistent iteration order in the removal loop below.
        live_set = set()
        live_list = []
        queue = deque()

        def _mark_live(inst):
            if inst not in live_set:
                live_set.add(inst)
                live_list.append(inst)
                queue.append(inst)

        for block in func.blocks:
            for inst in block.instructions:
                if inst.opcodes and inst.opcodes[0] in side_effect_instructions:
                    _mark_live(inst)

        while queue:
            inst = queue.popleft()
            for op, def_inst in inst.reaching_defs.items():
                _mark_live(def_inst)

        removed_insts = set()
        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                if inst not in live_set:
                    removed_insts.add(inst)
                else:
                    new_instructions.append(inst)

            block.instructions = new_instructions
