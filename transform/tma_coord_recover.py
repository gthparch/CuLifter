"""TmaCoordRecover — recover the 4D tensor coordinates feeding each UTMALDG.4D.

THE FLASHATTENTION BLOCKER (4D coords).  A cute `SM90_TMA_LOAD` lowers to
`UTMALDG.4D [smem], [desc_addr] [, desc[hint]]`.  The four tile coordinates
{c0,c1,c2,c3} are NOT printed operands and NOT encoded in the instruction word —
they are passed implicitly in a fixed uniform-register window per the cute TMA
ABI (verified across all 297 UTMALDG.4D in the real FA hdim64+hdim128 cubins and
an inline-PTX multitile microbench: the coords always live in UR10:UR11:UR12:
UR13, written by an `R2UR`/`UMOV`/`UIMAD`/`UIADD3` chain immediately before each
op; the varying coord, e.g. the seq-tile row, steps by the box size per loop
iteration).  Because they are not operands, SSA never versions them, DCE deletes
their build chain, and the lifter ran `utma_load` with coord=0 — so only tile
(0,0) of each Q/K/V tensor loaded and FA was numerically wrong.

This pass attaches the four coordinate registers as explicit USE operands of
each UTMALDG.4D.  UTMALDG has use_op_start_idx==0 (all operands are uses), so the
appended coord regs are immediately picked up by SSA (versioned to their correct
*reaching definition* — per-instruction dataflow, since each op's reaching def of
UR11 etc. is its own build chain), kept alive by DCE (UTMALDG is side-effecting
and traces its uses), and renamed by RegRemap.  The lifter then fetches the four
coord values through the normal operand path and passes them to `utma_load`.

Layering: this is a middle-end transform, not
a lifter patch. It
runs after BuildCFG (needs the CFG for cross-block reaching-def) and before SSA
(so the attached operands are versioned).  The coord window is *discovered* by
reaching-def — not hardcoded — so that an arch/kernel using a different window
is handled by the same dataflow rather than a magic constant.
"""
from transform.transform import SaSSTransform
from sir.operand import Operand


# Number of TMA tensor coordinates we attach (cute SM90_TMA_LOAD is rank-<=5;
# the .4D form carries 4).
_NUM_COORDS = 4

# The cute SM90_TMA_LOAD coordinate-register ABI window (low reg of the 4
# consecutive uniform regs).  Used as the discovery seed / fallback; the actual
# regs attached are whichever have a reaching def (verified UR10..UR13 for both
# real FA variants and the multitile microbench).
_ABI_COORD_LO = 10


class TmaCoordRecover(SaSSTransform):
    def apply(self, module):
        print("=== Start of TmaCoordRecover ===")
        for func in module.functions:
            self._process_func(func)
        print("=== End of TmaCoordRecover ===")

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _is_utmaldg_4d(inst):
        return (len(inst.opcodes) >= 2
                and inst.opcodes[0] == "UTMALDG"
                and "4D" in inst.opcodes)

    @staticmethod
    def _is_coord_target(inst):
        """UTMALDG.4D (load) or UTMASTG.Nd (store) — both carry implicit tile
        coordinates in the cute ABI uniform-register window. The FA epilogue's
        UTMASTG.5D re-stages the SAME coord regs (UR18=UR10, UR19=UR11,
        UR20=UR12 right before the store), so UR10..UR13 still reach the store
        with (headdim, seq, head) — attaching the canonical window recovers the
        output-tile seq/head so each tile's O lands in its own (seq,head) plane
        instead of every tile clobbering plane (0,0). (UR13=batch is zeroed at
        the store in this kernel; multi-batch O coverage is a separate item.)"""
        if not inst.opcodes:
            return False
        op0 = inst.opcodes[0]
        if op0 == "UTMALDG" and "4D" in inst.opcodes:
            return True
        if op0 == "UTMASTG":
            return True
        return False

    @staticmethod
    def _smem_dst_reg_num(inst):
        """The uniform-register NUMBER of the UTMALDG/UTMASTG shared operand
        (operand[0] = the `[URx]` smem address). The cute SM90_TMA_LOAD/STORE
        coordinate window is anchored at this register: ptxas passes the four
        tile coords {c0,c1,c2,c3} in the uniform regs immediately following the
        smem-dst register (smem+2 .. smem+5). The UTMALDG.4D instruction word
        encodes only three operand register fields (tmap, smem, desc) — there is
        NO field for a coord-window base — so the window is a fixed software-ABI
        offset from the smem register, NOT a magic UR10. Returns None if the
        shared operand isn't a `URn` mem-addr."""
        if not inst.operands:
            return None
        op0 = inst.operands[0]
        if not getattr(op0, "is_mem_addr", False):
            return None
        r = getattr(op0, "reg", None)
        if not r or not r.startswith("UR") or r == "URZ":
            return None
        try:
            return int(r[2:])
        except ValueError:
            return None

    @staticmethod
    def _written_uregs(inst):
        """Uniform registers defined by `inst` (the def operands)."""
        out = set()

        def _add(op):
            r = getattr(op, "reg", None)
            if r and r.startswith("UR") and r != "URZ":
                out.add(r)

        for op in inst.get_defs():
            if getattr(op, "is_reg_group", False):
                for sub in op.reg_group:
                    _add(sub)
            else:
                _add(op)
        return out

    def _reaching_defs_before(self, func, target, window=None):
        """Set of uniform-register names that have a reaching def on some path
        reaching `target` (intra-procedural).  Used to confirm a candidate
        coord reg is actually defined before the UTMALDG (so we don't attach a
        never-written register that SSA would then SETZERO).

        `window` (optional) is this load's own coord-register window. When the
        in-block defs already cover the window, we return them immediately —
        identical to the original in-block-first behaviour (so every load that
        passed before still passes byte-for-byte). Only when the in-block defs do
        NOT cover the window do we fall through to a cross-block predecessor BFS
        and UNION the predecessor defs in. This is what the FA3 BACKWARD dO load
        pair needs: the two sibling atoms issue from ONE software-pipelined loop
        block, but their coord windows are filled across the block boundary —
        the [UR16] atom's window (UR18..UR21) is written in-block while the
        [UR24] atom's window (UR26..UR29) is written in the PREDECESSOR block
        (UMOV UR26/UR28/UR29 + the loop-carried UR27). The old code returned the
        in-block set as soon as it was non-empty (it was, for the sibling's
        smem/UR18/UR19 builds), so the BFS never ran and the [UR24] window came
        back "no reaching def" -> coords defaulted to 0 -> that atom reloaded dO
        tile (0,0,0,0) every loop trip (STALE dO) while its sibling advanced ->
        dV = Sum_t P[trip t] . dO[stale block 0] -> uncorrelated. With the BFS
        union the [UR24] window's reaching def is found, TmaCoordRecover attaches
        it, and SSA versions the operand to its correct per-instruction reaching
        def (the live loop-carried m_base PHI), so the consumer GEMM reads the
        current trip's dO."""
        # Locate target's block + index.
        tgt_bb = None
        tgt_idx = None
        for bb in func.blocks:
            for i, ins in enumerate(bb.instructions):
                if ins is target:
                    tgt_bb, tgt_idx = bb, i
                    break
            if tgt_bb is not None:
                break
        if tgt_bb is None:
            return set()

        defined = set()
        # In-block defs before the target (most precise; the coord idiom is
        # usually in the same block right above the UTMALDG).
        for ins in tgt_bb.instructions[:tgt_idx]:
            defined |= self._written_uregs(ins)
        # Fast path (unchanged for the fwd kernel): the in-block defs already
        # cover this load's coord window -> no BFS, identical result/ordering.
        if defined and (window is None or any(r in defined for r in window)):
            return defined

        # Cross-block: BFS predecessors collecting any UR def (over-approx) and
        # UNION with the in-block defs.
        visited = {tgt_bb}
        stack = list(getattr(tgt_bb, "preds", []))
        while stack:
            bb = stack.pop()
            if bb in visited:
                continue
            visited.add(bb)
            for ins in bb.instructions:
                defined |= self._written_uregs(ins)
            stack.extend(getattr(bb, "preds", []))
        return defined

    # --------------------------------------------------------------- the pass
    def _process_func(self, func):
        targets = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if self._is_coord_target(inst):
                    targets.append((bb, inst))
        if not targets:
            return

        # The cute SM90_TMA_LOAD/STORE coordinate window is anchored PER
        # INSTRUCTION at (smem-dst register + 2): ptxas passes {c0,c1,c2,c3} in
        # the uniform regs immediately following the `[URx]` smem operand
        # (UR(x+2)..UR(x+5)). This holds for the verified-correct fwd kernel (smem
        # = UR8 -> UR10..UR13, == the old hardcode) AND for the bwd, where the
        # Q/dO load uses smem = UR16 -> UR18..UR21 and the K/V load uses smem =
        # UR8 -> UR10..UR13. The old hardcoded UR10..UR13 was only correct when
        # smem happened to be UR8; for the bwd rows=80 (Q/dO) load (smem = UR16)
        # it read UR10/UR11 = the `desc[UR10]` cache-hint pair {0, 0x14f00000},
        # fabricating c1 = 0x14f00000 (351272960) -> the TMA copied garbage into
        # the Q smem region and the S=Q.Kt GEMM read B=Q all-zero. Deriving the
        # window from the smem register (the instruction word has no coord-window
        # field, only tmap/smem/desc reg fields) recovers each load's real
        # {c0,c1,c2,c3}.
        #
        # UTMASTG keeps the _ABI_COORD_LO (UR10..UR13) seed: the cute
        # SM90_TMA_STORE epilogue re-stages every atom's coords into UR10..UR13
        # (verified — the fwd O store uses smem [UR8] AND [UR16] but BOTH atoms
        # read UR10..UR13). Applying the smem+2 rule to stores would shift the
        # [UR16] atom to UR18..UR21 and break the fwd byte-identity.
        def _window_for(inst):
            if inst.opcodes and inst.opcodes[0] == "UTMALDG":
                n = self._smem_dst_reg_num(inst)
                lo = (n + 2) if n is not None else _ABI_COORD_LO
            else:
                lo = _ABI_COORD_LO
            return [f"UR{lo + i}" for i in range(_NUM_COORDS)]

        # Pass 1: which targets have THEIR OWN coord window reaching them, and
        # which (block, window) pairs that proves valid.
        #
        # A cute SM90_TMA_LOAD for one tile lowers to a *pair* of UTMALDG.4D
        # (one per 64-elem headdim atom) sharing the same {c1,c2,c3} tile coords
        # — only c0 (headdim) differs (0 then 0x40). Each atom writes its OWN
        # smem-dst register, so each atom has its OWN window (atom0 smem=UR8 ->
        # UR10.., atom1 smem=UR40 -> UR42..), and that window's first reg already
        # carries the per-atom c0. A coord-target whose own window has a reaching
        # def proves that window for any same-block sibling that shares the SAME
        # window registers (e.g. predicated coord builds: atom0 issued before the
        # in-block c0 bump still resolves to the dominating coord PHIs).
        self_present = {}
        block_window_ok = set()   # (id(bb), window-tuple) proven valid
        for bb, inst in targets:
            if getattr(inst, "_tma_coords_attached", False):
                continue
            window = _window_for(inst)
            reaching = self._reaching_defs_before(func, inst, window)
            present = any(r in reaching for r in window)
            self_present[id(inst)] = present
            if present:
                block_window_ok.add((id(bb), tuple(window)))

        # Pass 2: attach each target's OWN window if its own probe found it, or a
        # same-block sibling proved the identical window.
        attached = 0
        for bb, inst in targets:
            # Already has coord operands attached (re-run guard).
            if getattr(inst, "_tma_coords_attached", False):
                continue
            window = _window_for(inst)
            if not (self_present.get(id(inst), False)
                    or (id(bb), tuple(window)) in block_window_ok):
                # No coord regs reach this op and no same-block sibling confirms
                # this window — leave it; lifter falls back to coord=0.
                continue

            # Append the 4 coord registers as use operands (tag so the lifter
            # can identify them and so a re-run is idempotent).
            for r in window:
                op = Operand.from_reg(r, r)
                op.is_tma_coord = True
                op.Parent = inst
                inst.operands.append(op)
            inst._tma_coords_attached = True
            attached += 1

        if attached:
            print(f"  {func.name}: attached coords to {attached} UTMALDG.4D/UTMASTG")
