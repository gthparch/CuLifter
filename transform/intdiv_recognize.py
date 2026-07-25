"""Recognise ptxas's magic-multiplication runtime int-division SASS pattern
and replace its quotient-producing instruction with a synthetic SDIV op.

The pattern is anchored by `IADD3 R, R_rcp, 0xffffffe, RZ` — the constant
0x0FFFFFFE is unique to ptxas's reciprocal-magic lowering. From the anchor
we walk back through MUFU.RCP to the I2F.RP whose source is the divisor
MAGNITUDE (the IABS output for a signed C divide, the raw value for an
unsigned one), then forward to the *leaf* IMAD.HI.U32 instruction(s) —
each a pre-correction quotient `hi(magic * |dividend|)` — and replace each
with `SDIV q, |dividend|, |divisor|`.

SDIV's operands are deliberately the UNSIGNED MAGNITUDES the chain itself
consumes (I2F's source, the leaf's second multiplicand), NOT the original
signed values behind the IABS instructions: the leaf computes
`hi(magic * |n|)` ~= `|n| / |d|`, and the downstream correction
(`rem = |n| - q*|d|`, conditional +1) and sign fixup
(`@((n^d)<0) q = -q`) only work when the def holds exactly `|n| /u |d|`.
Splicing the signed values broke every negative-operand divide
(idivide-cuda, H200 pilot 142532: `33664 / -1` → udiv(33664, 0xFFFFFFFF)
= 0 → two corrections → 2 → sign fixup → -2); the IABS-trace was a
leftover from the abandoned design that lifted SDIV as a signed `sdiv`
handed back to ptxas.

On sm90, ptxas may promote the magic into UNIFORM registers (cudnn so.69's
warpid helper, `x / warpSize`): the magic hops `F2I → R2UR` into a UR, the
Newton refinement becomes `UIMAD.WIDE.U32` with the magic in the
addend-pair-HI trick, and a uniform quotient (`blockDim/32`) is consumed as
the HI WORD of a `UIMAD.WIDE.U32` dest pair instead of an IMAD.HI def. The
walk therefore traverses value-preserving copies (R2UR/MOV/UMOV) like PHIs,
treats `(U)IMAD.WIDE.U32` as chain nodes, and rewrites WIDE leaves by
defining just the HI reg (guarded: zero addend, dead LO word, register
dividend).

Because the walk treats `IMAD.WIDE.U32` as a chain node, a *vector* `IMAD.HI`
quotient leaf whose result is later used as an address-index multiplicand
(`off = q*stride+base`, RowMajor) would be walked PAST: the quotient flows
through the Granlund-Möller +1 correction's SSA join into that address WIDE,
which the walk mistakes for a further chain node (then the leaf-rewrite guard
rejects it — non-zero base addend, live LO word — and the divide is lost). The
distinguishing signal is the join PHI itself: the correction merges `q` with
`q+1`, whereas a genuine inter-refinement PHI merges two reciprocal-refinement
values. `_is_quotient_correction_phi` detects the `q`/`q+1` shape and stops the
hop there, so the IMAD.HI quotient is recognised as the leaf (45_dual_gemm's
grid-strided `coord = index/N`; the `q`-only `q[i]=i/N` form is unaffected —
there the address WIDE multiplies the store index, not the quotient).

A second, 64-bit flavour of the same idiom (ptxas's u64 divide/modulo
helper: `I2F.U64.RP → MUFU.RCP → IADD3 +0x1FFFFFFE → F2I.U64.TRUNC` followed
by a Newton-Raphson refinement of the 64-bit reciprocal) is recognised by
`_try_rewrite_u64` and cut at the pre-correction REMAINDER pair, which is
replaced with `UREM64`/`UREM64HI` pseudo-ops (lifted as a div-by-zero-guarded
`urem i64`). See that method's docstring for the slicing details.

A plain divide has one leaf. A combined divide+modulo (`a/d` and `a%d`)
shares one magic number that fans out to two leaves (one feeds the quotient,
one feeds `dividend - q*d` for the remainder); both must be rewritten or the
un-rewritten one keeps the hardware-reciprocal magic and stays wrong. See
`_walk_imad_hi_leaves`.

Why replace inside the chain: lifting I2F.RP / MUFU.RCP / F2I.U32.TRUNC
faithfully via LLVM intrinsics produces slightly different bit patterns
than the SASS hardware ops, and ptxas's correction-step predicates are
bit-tuned for the exact hardware MUFU.RCP — small ULP differences cascade
into off-by-one quotients (on the CPU backend's software rcp_f32 they are
wholly wrong). The IMAD.HI.U32 def is a clean cut point: the downstream
correction (R0 = d*(-q) + |t|, ISETP.GT, @!P IADD3 q+=1) reads only the
quotient and divisor/dividend, so when q is already correct from sdiv the
predicate skips the correction and the value flows through unchanged. The
remainder leaf recovers `dividend - q*d` from its own (now exact) quotient,
so modulo works without any extra opcode.

Registered in both lifter pipelines (lifter_nvvm.py and lifter_cpu.py).
"""
from sir.instruction import Instruction
from transform.transform import SaSSTransform


_MAGIC_CONSTANT = 0x0FFFFFFE
# The u64 divide/modulo chain biases the reciprocal with 0x1FFFFFFE instead
# (one extra exponent step — the estimate seeds a 64-bit Newton-Raphson).
_MAGIC_CONSTANT_U64 = 0x1FFFFFFE


class IntDivRecognize(SaSSTransform):
    def apply(self, module):
        super().apply(module)
        # changed=False lets the pipeline skip the def-use rebuild that
        # follows this pass when no divide chain was rewritten (the common
        # case — most modules have no magic-divide sites).
        self.changed = False
        for func in module.functions:
            if not func.blocks:
                continue
            # Lazy per-function dominator tree for the splice soundness
            # check (BUG-092) — only built when a magic anchor is found.
            self._idom = None
            self._idom_func = func
            for block in func.blocks:
                # Snapshot — the rewrites splice block.instructions.
                for inst in list(block.instructions):
                    if self._is_magic_anchor(inst, _MAGIC_CONSTANT):
                        if self._try_rewrite(block, inst):
                            self.changed = True
                    elif self._is_magic_anchor(inst, _MAGIC_CONSTANT_U64):
                        if self._try_rewrite_u64(inst):
                            self.changed = True

    def _dominates(self, a, b):
        """True iff block a dominates block b (lazy idom walk)."""
        if a is b:
            return True
        if self._idom is None:
            from transform.domtree import DominatorTree
            f = self._idom_func
            self._idom = DominatorTree(f.blocks[0], f.blocks).idom
        idom = self._idom
        cur = b
        while True:
            nxt = idom.get(cur)
            if nxt is None or nxt is cur:
                return False
            if nxt is a:
                return True
            cur = nxt

    def _phi_reaching_version(self, def_inst, target_block):
        """BUG-092: find the SSA version carrying `def_inst`'s value at
        `target_block` when def_inst's own block does not dominate it.

        The chain walked to the leaf through CFG joins (PHI hops in
        _imad_hi_users_through_phi), so the value's live version at the
        leaf is a PHI merge. BFS def_inst's users through PHI nodes only
        (the same philosophy as the leaf walk: arithmetic is never
        traversed), collecting PHIs whose block dominates target_block.
        Returns the deepest-dominating candidate (the most-merged version,
        nearest the leaf), or None — sorted tie-break keeps it
        deterministic. Depth = idom-chain length, computed lazily."""
        cands = []
        seen = {id(def_inst)}
        pending = [def_inst]
        while pending:
            node = pending.pop()
            for user, _op in self._users(node):
                if (user.opcodes and user.opcodes[0] == "PHI"
                        and id(user) not in seen):
                    seen.add(id(user))
                    pending.append(user)
                    if user.parent is not None and self._dominates(
                            user.parent, target_block):
                        cands.append(user)
        if not cands:
            return None

        def depth(blk):
            d = 0
            cur = blk
            while True:
                nxt = self._idom.get(cur)
                if nxt is None or nxt is cur:
                    return d
                d += 1
                cur = nxt

        cands.sort(key=lambda p: (-depth(p.parent), str(p.id)))
        return cands[0]

    def _resolve_splice_operands(self, target_block, pairs):
        """BUG-092: the spliced SDIV introduces NEW uses of the
        dividend/divisor registers at the leaf — a point the original
        program reached only through the chain's PHI-mediated dataflow.
        This pass runs post-SSA, so an operand is usable as-is only when
        its def block dominates the leaf block. When it does not, no
        direct placement is sound (dom(def,P) and dom(P,leaf) would imply
        dom(def,leaf)) — instead substitute the PHI-mediated version live
        at the leaf: SSA placed PHIs merging the operand's versions
        (spill reloads of one variable, predicated redefinitions) at the
        joins the chain crossed, so the deepest dominating merge IS the
        value the original chain consumed there (so.69/so.75 bn kernels:
        LDL reload feeding a leaf in a PBRA-split join block).

        Returns [dividend_use, divisor_use] operand clones ready for the
        SDIV, or None when some operand has no version live at the leaf
        (then the caller skips loudly and the chain lifts faithfully).
        pairs = (operand, instruction whose reaching_defs resolves it)."""
        import os
        dbg = os.environ.get("SLIFTER_INTDIV_DEBUG") == "1"
        resolved = []
        for op, src in pairs:
            if op is None or not getattr(op, "is_reg", False):
                # immediates / const-mem operands are position-independent
                resolved.append(op.clone() if op is not None else None)
                continue
            d = (src.reaching_defs.get(op)
                 if src is not None and hasattr(src, "reaching_defs") else None)
            if d is None or d.parent is None:
                return None  # unknown def: be conservative
            if self._dominates(d.parent, target_block):
                resolved.append(op.clone())
                continue
            phi = self._phi_reaching_version(d, target_block)
            if phi is None:
                if dbg:
                    self._dump_unsound(op, d, target_block)
                return None
            sub = phi.get_defs()[0].clone()
            if dbg:
                import builtins
                import functools
                import sys
                eprint = functools.partial(builtins.print, file=sys.stderr)
                eprint(f"[intdiv-dbg] substituted {op} (def {d.id} does not "
                       f"dominate leaf) -> PHI-mediated {sub} from {phi.id}",
                       flush=True)
            resolved.append(sub)
        return resolved

    def _dump_unsound(self, op, def_inst, target_block):
        """SLIFTER_INTDIV_DEBUG=1 forensics for guard skips (BUG-092):
        report the non-dominating operand and whether a PHI-mediated
        version live at the leaf exists — the data that decides if the
        reaching-version substitution is sound for this shape."""
        import builtins
        import functools
        import sys
        # stderr — the per-function pipeline nulls stdout in non-verbose
        # runs (lifter._transform_per_function), same as _try_rewrite_u64.
        print = functools.partial(builtins.print, file=sys.stderr)
        bname = lambda b: getattr(b, "name", None) or getattr(
            b, "addr_content", b)
        print(f"[intdiv-dbg] unsound operand {op} "
              f"def={def_inst.opcodes}@{def_inst.id} "
              f"defblk={bname(def_inst.parent)} "
              f"leafblk={bname(target_block)}", flush=True)
        phi = self._phi_reaching_version(def_inst, target_block)
        if phi is None:
            print("[intdiv-dbg]   no PHI-mediated version dominates the "
                  "leaf — substitution impossible for this shape", flush=True)
            return
        ins = []
        for u in phi.get_uses():
            dd = phi.reaching_defs.get(u) if hasattr(phi, "reaching_defs") \
                else None
            ins.append(f"{u}<-{dd.opcodes if dd else '?'}@"
                       f"{dd.id if dd else '?'}")
        print(f"[intdiv-dbg]   candidate PHI {phi.id} "
              f"blk={bname(phi.parent)} def={phi.get_defs()[0]} "
              f"inputs: {' | '.join(ins)}", flush=True)

    @staticmethod
    def _is_magic_anchor(inst, magic):
        # ptxas emits either `IADD3 R, R_rcp, <magic>, RZ` or
        # `VIADD R, R_rcp, <magic>` — same semantic (`R_rcp + magic`),
        # different opcode. The walk-back uses get_uses()[0] = the first
        # register operand (R_rcp), which is identical in both forms.
        if not inst.opcodes or inst.opcodes[0] not in ("IADD3", "VIADD"):
            return False
        return any(op.is_immediate and op.immediate_value == magic
                   for op in inst.operands)

    @staticmethod
    def _walk_back(inst, use_op, expect_opcode, expect_modifier=None):
        def_inst = inst.reaching_defs.get(use_op)
        if def_inst is None or not def_inst.opcodes:
            return None
        if def_inst.opcodes[0] != expect_opcode:
            return None
        if expect_modifier is not None and (
                len(def_inst.opcodes) < 2
                or def_inst.opcodes[1] != expect_modifier):
            return None
        return def_inst

    @staticmethod
    def _users(inst):
        out = []
        for use_set in inst.users.values():
            out.extend(use_set)
        return out

    @staticmethod
    def _find_user(inst, expect_opcode):
        for use_inst, _ in IntDivRecognize._users(inst):
            if use_inst.opcodes and use_inst.opcodes[0] == expect_opcode:
                return use_inst
        return None

    @staticmethod
    def _is_mul_operand_use(user, use_op):
        """True iff this def-use edge feeds one of `user`'s multiplicand
        operands (uses[0]/uses[1]).  A magic-divide chain propagates only
        through multiply inputs.  sm70+ IMAD.HI register addends are 64-bit
        pairs (sir/instruction.py); their sub-operand use edges are genuine
        data dependences but NOT chain hops — following them bridges into
        unrelated chains and double-rewrites a shared leaf (car-cuda:
        `.index(leaf)` ValueError after the first rewrite spliced it out)."""
        uses = user.get_uses()
        return len(uses) >= 2 and (use_op is uses[0] or use_op is uses[1])

    # Value-preserving hops the chain walk traverses transparently: PHI
    # joins, plus the copies the sm90 uniform-promoted variant routes the
    # magic through (R2UR vector→uniform; MOV/UMOV plain moves — the walk
    # runs before MovEliminate). Predicated copies are merges, not copies,
    # and are never hopped.
    _HOP_OPS = ("PHI", "R2UR", "MOV", "UMOV")

    @staticmethod
    def _is_chain_node(user):
        """True iff `user` can carry the magic-multiply chain in its def:
        the classic vector `IMAD.HI[.U32]` (refinement and leaves), or the
        sm90 uniform-promoted `(U)IMAD.WIDE.U32` whose dest-pair HI word is
        the chain value (cudnn so.69 warpid helper: both the Newton
        refinement — magic in the addend-pair-HI trick — and the uniform
        quotient leaf appear as UIMAD.WIDE.U32). Signed `IMAD.WIDE` (no
        .U32) is address arithmetic, never a chain hop."""
        if not user.opcodes or user.opcodes[0] not in ("IMAD", "UIMAD"):
            return False
        if len(user.opcodes) < 2:
            return False
        if user.opcodes[1] == "HI":
            return True
        return user.opcodes[1] == "WIDE" and "U32" in user.opcodes[2:]

    @classmethod
    def _correction_increment_source(cls, inst):
        """If `inst` computes `src + small_immediate` (the conditional `q+=1`
        the magic-divide correction emits after the quotient leaf), return the
        register operand `src` being incremented; else None. Accepts IADD3 /
        VIADD / IADD with an immediate operand, and IMAD.IADD (`src*1 + imm`,
        ptxas's fused form). A predicate-guarded def is fine — the +1 fixup is
        itself predicated."""
        if not inst.opcodes:
            return None
        op0 = inst.opcodes[0]
        if op0 == "IMAD":
            if len(inst.opcodes) < 2 or inst.opcodes[1] != "IADD":
                return None
        elif op0 not in ("IADD3", "VIADD", "IADD"):
            return None
        uses = inst.get_uses()
        if not any(u.is_immediate for u in uses):
            return None
        regs = [u for u in uses
                if u.is_reg and not u.is_rz and not u.is_immediate
                and not u.is_predicate_reg and not u.is_not_reg]
        return regs[0] if regs else None

    @classmethod
    def _is_quotient_correction_phi(cls, phi):
        """True iff `phi` merges a chain value `v` with `v + small_imm` — the
        SSA join the Granlund-Möller quotient correction (`@P q += 1`) produces
        after the magic leaf. Such a PHI carries the FINALISED quotient, not a
        refinement value, so the walk must NOT hop through it: doing so descends
        past the true IMAD.HI quotient leaf into the downstream consumers of the
        quotient (e.g. the 64-bit `IMAD.WIDE.U32 q*stride+base` row-major address
        in 45_dual_gemm's grid-strided coord = index/N), which the leaf-rewrite
        guard then rejects (non-zero base addend, live LO word) — leaving the
        divide unrecognised and the chain lifted faithfully (off-by-one). The
        genuine inter-refinement PHIs (mr-cuda's compute_modn32) merge two
        reciprocal-refinement values, NOT `v`/`v+imm`, so they are unaffected.

        Detection uses instruction identity (a `v+imm` input whose incremented
        register's reaching-def is itself another input of the same PHI), never
        register-name string matching, so SSA versioning can't spoof it."""
        if not (phi.opcodes and phi.opcodes[0] == "PHI"):
            return False
        input_defs = [phi.reaching_defs.get(u)
                      for u in phi.get_uses()] if hasattr(
            phi, "reaching_defs") else []
        input_ids = {id(d) for d in input_defs if d is not None}
        if not input_ids:
            return False
        for d in input_defs:
            if d is None:
                continue
            src = cls._correction_increment_source(d)
            if src is None:
                continue
            src_def = d.reaching_defs.get(src) if hasattr(
                d, "reaching_defs") else None
            if src_def is not None and id(src_def) in input_ids:
                return True
        return False

    @classmethod
    def _chain_users_through_hops(cls, inst):
        """Return (chain_node, use_op) for all chain nodes reachable from
        `inst` through zero or more transparent hops (_HOP_OPS).  A direct
        chain-node user is returned immediately; a hop user is expanded to
        its own chain-node (and further-hop) users.  Only hops are
        followed — arithmetic instructions are not traversed, so the walk
        cannot escape into unrelated code that happens to read a hop
        output. A `seen` set guards against cycles.  Chain-node hops count
        only when the value feeds a multiplicand — addend-pair edges (see
        _is_mul_operand_use) are excluded."""
        result = []
        hop_seen = set()
        pending = [inst]
        while pending:
            node = pending.pop()
            for user, use_op in cls._users(node):
                if not user.opcodes:
                    continue
                if cls._is_chain_node(user):
                    if cls._is_mul_operand_use(user, use_op):
                        result.append((user, use_op))
                elif user.opcodes[0] in cls._HOP_OPS:
                    if user.opcodes[0] != "PHI" and user.pflag is not None:
                        continue
                    # A quotient-correction PHI (merges q with q+1) carries the
                    # finalised quotient, not a refinement value — hopping it
                    # would walk past the true IMAD.HI quotient leaf into the
                    # quotient's downstream address arithmetic (dual_gemm
                    # IMAD.WIDE.U32 q*stride+base). Stop here so the node feeding
                    # this PHI is recognised as the leaf.
                    if (user.opcodes[0] == "PHI"
                            and cls._is_quotient_correction_phi(user)):
                        continue
                    if id(user) not in hop_seen:
                        hop_seen.add(id(user))
                        pending.append(user)
        return result

    @classmethod
    def _walk_imad_hi_leaves(cls, start):
        """Collect every *leaf* chain node reachable from `start` along
        chain def-use edges — a node that computes `hi(magic * dividend)`
        and feeds no further chain node. The refinement chain (F2I → … →
        magic number) is walked through; the magic number's chain-node
        users are the leaves. Returns (leaf, chain_use_op) pairs — the
        recorded edge identifies WHICH multiplicand the magic feeds, so the
        rewriter can pick the dividend as the other one (classic ptxas puts
        the magic at uses[0]; the sm90 uniform variant at uses[1]).

        PHI nodes (and R2UR/MOV/UMOV copies, see _HOP_OPS) between two
        chain nodes are traversed transparently: a node whose result
        reaches another chain node through one or more hops is NOT a leaf;
        the leaf is the final node on the other side. This handles the case
        where an unpredicated instruction between two predicated IMAD.HIs
        causes BuildCFG to split them into separate PBRA blocks, forcing
        SSA to insert a PHI at the join point (mr-cuda's compute_modn32:
        0x01d0 refinement → PHI → 0x01f0 true leaf), and the sm90
        uniform-promoted magic (F2I → R2UR → UIMAD.WIDE.U32 refinement).

        A plain divide has one leaf. A combined divide+modulo (`a/d` and
        `a%d`, as in nvjpeg's per-image index + in-image position) shares one
        magic number that fans out to two leaves — both must be rewritten, or
        the un-rewritten one keeps the hardware-reciprocal magic and is wrong
        on the CPU backend. Returns [] if `start` has no chain-node user."""
        leaves = []
        seen = set()
        frontier = list(cls._chain_users_through_hops(start))
        while frontier:
            cur, use_op = frontier.pop()
            if id(cur) in seen:
                continue
            seen.add(id(cur))
            nxt = cls._chain_users_through_hops(cur)
            if nxt:
                frontier.extend(nxt)
            else:
                leaves.append((cur, use_op))
        return leaves

    def _try_rewrite(self, block, anchor):
        # Walk back: anchor → MUFU.RCP → I2F. I2F's source IS the divisor
        # magnitude |d| (the IABS output for a signed C divide, the raw
        # divisor when ptxas knew it non-negative) — exactly the value the
        # chain divides by, so it is what SDIV must udiv by. Do NOT trace
        # through the IABS to the signed original: the downstream
        # correction + sign-fixup steps assume the def is |n| /u |d|.
        anchor_uses = anchor.get_uses()
        if not anchor_uses:
            return
        rcp = self._walk_back(anchor, anchor_uses[0], "MUFU", "RCP")
        if rcp is None:
            return
        i2f = self._walk_back(rcp, rcp.get_uses()[0], "I2F")
        if i2f is None:
            return
        i2f_uses = i2f.get_uses()
        divisor_op = i2f_uses[0]

        # Walk forward: anchor → F2I → magic-refinement chain → leaf IMAD.HI(s).
        # Each leaf is a pre-correction quotient `hi(magic * dividend)`; divide
        # and modulo share the magic number but have separate leaves.
        f2i = self._find_user(anchor, "F2I")
        if f2i is None:
            return
        leaves = self._walk_imad_hi_leaves(f2i)
        if not leaves:
            return

        rewrote = False
        for leaf, chain_op in leaves:
            # The dividend is the multiplicand the chain does NOT feed —
            # the magnitude |n| the chain multiplies by the magic number
            # (classic ptxas puts the magic at uses[0] and |n| at uses[1];
            # the sm90 uniform variant swaps them). Use it as-is (it is
            # already an operand AT the leaf, so its def trivially
            # dominates the splice point); see module docstring for why
            # the signed original behind the IABS is wrong here.
            leaf_uses = leaf.get_uses()
            if len(leaf_uses) < 2:
                continue
            dividend_op = (leaf_uses[1] if chain_op is leaf_uses[0]
                           else leaf_uses[0])

            # Replace the leaf IMAD.HI.U32 directly. Its def is the
            # pre-correction quotient register; the downstream correction step
            # (and, for modulo, the `dividend - q*divisor` recovery) reads only
            # this def + dividend + divisor, so feeding it the exact quotient
            # makes the correction predicate skip and the right value flow
            # through. The leaf may live in a different block than the anchor
            # (def-use chains cross basic blocks) — splice into its own block.
            target_block = leaf.parent
            try:
                leaf_idx = target_block.instructions.index(leaf)
            except ValueError:
                # Two magic anchors can share one refinement chain (the SASS
                # computes the magic once and both IADD3 candidates walk to
                # the same leaves); the first anchor already replaced this
                # leaf with the identical SDIV (dividend is leaf-derived,
                # divisor chain-derived), so there is nothing left to do.
                continue

            if leaf.opcodes[1] == "WIDE":
                # sm90 uniform-promoted leaf: `UIMAD.WIDE.U32 {lo,hi}, m,
                # n, 0` — the quotient is the dest pair's HI word. The
                # rewrite (SDIV defines just the hi reg) is sound only when
                # the addend is zero (else the hi word isn't plain
                # hi(m*n)), the LO word is dead (so.69: overwritten by the
                # correction's rem before any read), and the dividend is a
                # register (an immediate multiplicand means this WIDE is
                # address arithmetic the walk mistook for a leaf). Skip
                # loudly otherwise — the chain then lifts faithfully.
                addend = leaf_uses[2] if len(leaf_uses) > 2 else None
                addend_zero = addend is not None and (
                    bool(addend.reg_group
                         and all(s.is_rz for s in addend.reg_group))
                    or (addend.is_immediate and addend.immediate_value == 0))
                dest_pair = leaf.get_defs()[0]
                lo_sub = (dest_pair.reg_group[0]
                          if dest_pair.reg_group else None)
                lo_dead = lo_sub is not None and not leaf.users.get(lo_sub)
                if not (addend_zero and lo_dead and dividend_op.is_reg):
                    import sys
                    print(f"[intdiv] skipping WIDE leaf {leaf.id}: "
                          f"addend_zero={addend_zero} lo_dead={lo_dead} "
                          f"reg_dividend={dividend_op.is_reg}; "
                          f"chain lifts faithfully",
                          file=sys.stderr, flush=True)
                    continue
                dest_op = dest_pair.reg_group[1].clone()
            else:
                # A plain quotient leaf is exactly `IMAD.HI q = n, magic, RZ`:
                # ONE def and a zero addend. The PHI-transparent walk can
                # reach IMAD.HIs inside u64 multiply idioms — car-cuda's
                # `IMAD.HI.U32 R8, P0, R7, R4, R8` carries an addend AND a
                # carry-out predicate; replacing it with SDIV both drops the
                # P0 def (dangling IMAD.X carry-in downstream → lift
                # KeyError) and substitutes a quotient for a partial sum.
                leaf_defs = leaf.get_defs()
                addend = leaf_uses[2] if len(leaf_uses) > 2 else None
                addend_zero = addend is not None and (
                    addend.is_rz
                    or bool(addend.reg_group
                            and all(s.is_rz for s in addend.reg_group))
                    or (addend.is_immediate and addend.immediate_value == 0))
                if len(leaf_defs) != 1 or not addend_zero:
                    import sys
                    print(f"[intdiv] skipping non-quotient leaf {leaf.id}: "
                          f"defs={len(leaf_defs)} addend_zero={addend_zero}; "
                          f"chain lifts faithfully",
                          file=sys.stderr, flush=True)
                    continue
                dest_op = leaf_defs[0].clone()

            splice_ops = self._resolve_splice_operands(
                target_block,
                [(dividend_op, leaf), (divisor_op, i2f)])
            if splice_ops is None:
                import sys
                print(f"[intdiv] BUG-092 guard: skipping SDIV splice at "
                      f"{leaf.id} (operand def does not dominate leaf block "
                      f"{target_block.name if hasattr(target_block, 'name') else target_block} "
                      f"and no PHI-mediated version is live there); "
                      f"chain lifts faithfully",
                      file=sys.stderr, flush=True)
                continue
            new_inst = Instruction(
                id=f"{anchor.id}_intdiv_{leaf.id}",
                opcodes=["SDIV"],
                operands=[dest_op] + splice_ops,
                parentBB=target_block,
            )
            new_inst.use_op_start_idx = 1
            target_block.instructions[leaf_idx] = new_inst
            rewrote = True
        # Intermediate magic-mul instructions become unused; DCE removes
        # them in the next pipeline iteration.
        return rewrote

    # ------------------------------------------------------------------
    # 64-bit divide/modulo chain (ptxas's software u64 div helper)
    # ------------------------------------------------------------------

    @staticmethod
    def _forward_slice(start):
        """All instructions data-dependent (transitively, via SSA def-use
        `users` edges) on `start`'s defs, CUT AT PHIs. The NR refinement /
        quotient / remainder chain is straight-line dataflow; a PHI means a
        CFG merge, i.e. we left this chain's arithmetic. Without the cut,
        a chain inside a loop (mr-cuda's modular-exponentiation `% n`) has
        its slice wrap the back edge through the loop-head PHIs and swallow
        the def of its own dividend, breaking the in/out-of-slice tests.
        Returns (set of `id(inst)`, list of the instructions themselves)."""
        slice_ids = set()
        slice_insts = []
        frontier = [u for u, _ in IntDivRecognize._users(start)]
        while frontier:
            cur = frontier.pop()
            if id(cur) in slice_ids:
                continue
            if cur.opcodes and cur.opcodes[0] == "PHI":
                continue
            slice_ids.add(id(cur))
            slice_insts.append(cur)
            frontier.extend(u for u, _ in IntDivRecognize._users(cur))
        return slice_ids, slice_insts

    @staticmethod
    def _def_in_slice(inst, use_op, slice_ids):
        d = inst.reaching_defs.get(use_op)
        return d is not None and id(d) in slice_ids

    @classmethod
    def _match_rem_lo(cls, inst, slice_ids):
        """A = `IADD3 rem_lo, P, -X, Dlo, RZ` — the pre-correction remainder's
        low word: `Dlo - lo(q*d)`, with the borrow in P. X (the q*d low word)
        is inside the NR slice; Dlo (the dividend low word) is not. Returns
        the Dlo operand, or None if `inst` doesn't match."""
        if not inst.opcodes or inst.opcodes[0] != "IADD3":
            return None
        if any(m == "X" for m in inst.opcodes[1:]):
            return None
        defs = inst.get_defs()
        if len(defs) != 2 or not defs[1].is_predicate_reg or defs[1].is_pt:
            return None
        uses = inst.get_uses()
        if len(uses) != 3:
            return None
        neg = [u for u in uses if u.is_reg and u.is_negative_reg and not u.is_rz]
        rz = [u for u in uses if u.is_rz]
        plain = [u for u in uses
                 if u.is_reg and not u.is_rz and not u.is_negative_reg
                 and not u.is_not_reg and not u.is_predicate_reg]
        if len(neg) != 1 or len(rz) != 1 or len(plain) != 1:
            return None
        if not cls._def_in_slice(inst, neg[0], slice_ids):
            return None
        if cls._def_in_slice(inst, plain[0], slice_ids):
            return None
        return plain[0]

    @classmethod
    def _match_rem_hi(cls, inst, slice_ids, carry_def):
        """B = the pre-correction remainder's HIGH word, `Dhi - hi(q*d) - borrow`
        (== Dhi + ~hi(q*d) + borrow), where `carry_def`'s borrow predicate P is
        the carry-in. `~Y = ~hi(q*d)` is inside the NR slice; Dhi (dividend high
        word) is not. ptxas emits this high-word subtract-with-borrow in two
        interchangeable opcode forms — both must be recognised or the whole u64
        divide is lost:
          IMAD.X  rem_hi, Dhi, 0x1, ~Y, P        (B_u64mod's divide chain)
          IADD3.X rem_hi, ~Y, Dhi, RZ, P, !PT    (45_dual_gemm's modulo chain)
        Returns the Dhi operand, or None.  Matched by ROLE (operand order is not
        fixed across the two forms), mirroring `_match_rem_lo`."""
        if not inst.opcodes or not any(m == "X" for m in inst.opcodes[1:]):
            return None

        def _is_dhi(op):
            # Dividend high word: a plain register (not RZ / negated / ~ / pred),
            # defined OUTSIDE the NR slice.
            if (not op.is_reg or op.is_rz or op.is_negative_reg
                    or op.is_not_reg or op.is_predicate_reg):
                return False
            return not cls._def_in_slice(inst, op, slice_ids)

        def _is_not_y(op):
            # ~hi(q*d): a complemented register whose def is INSIDE the slice.
            return (op.is_reg and op.is_not_reg
                    and cls._def_in_slice(inst, op, slice_ids))

        def _is_borrow_pred(op):
            # The carry-in predicate == A's borrow def (not !P, not PT).
            return (op.is_predicate_reg and not op.is_not_reg and not op.is_pt
                    and inst.reaching_defs.get(op) is carry_def)

        uses = inst.get_uses()

        if inst.opcodes[0] == "IMAD":
            # Rigid IMAD.X layout: [Dhi, 0x1, ~Y, P].
            if len(uses) != 4:
                return None
            dhi, one, not_y, pred = uses
            if not (one.is_immediate and one.immediate_value == 1):
                return None
            if not _is_not_y(not_y) or not _is_borrow_pred(pred):
                return None
            if not _is_dhi(dhi):
                return None
            return dhi

        if inst.opcodes[0] == "IADD3":
            # IADD3.X form: exactly one ~Y (not-reg in-slice), exactly one Dhi
            # (plain reg out-of-slice), and the borrow-in predicate == carry_def.
            # Remaining operands are RZ / !PT fillers. Role-based, order-free.
            if not any(_is_borrow_pred(u) for u in uses):
                return None
            not_ys = [u for u in uses if _is_not_y(u)]
            dhis = [u for u in uses if _is_dhi(u)]
            if len(not_ys) != 1 or len(dhis) != 1:
                return None
            return dhis[0]

        return None

    def _try_rewrite_u64(self, anchor):
        """Recognise the u64 divide/modulo chain and replace its
        pre-correction remainder pair with UREM64/UREM64HI pseudo-ops.

        Backward from the anchor (`IADD3 R, R_rcp, 0x1FFFFFFE, RZ`):
        MUFU.RCP ← I2F.U64.RP, whose reg-pair source is the 64-bit DIVISOR.
        Forward: the anchor feeds F2I.U64.TRUNC (the u64 reciprocal seed);
        its transitive def-use slice is the Newton-Raphson refinement plus
        the quotient/remainder computation. Inside that slice the
        pre-correction remainder `dividend - q*divisor` is computed as the
        instruction pair
            A: IADD3  rem_lo, P, -X, Dlo, RZ      (X = lo(q*d), in slice)
            B: IMAD.X rem_hi, Dhi, 0x1, ~Y, P     (Y = hi(q*d), in slice)
        with Dlo/Dhi (the dividend words) defined OUTSIDE the slice and P
        consumed only by B. q comes from the truncated-rcp estimate, so the
        SASS remainder can be off by a couple of multiples of the divisor —
        the downstream conditional-subtract correction reads only rem/d.
        Feeding it the EXACT remainder (urem i64) makes every correction
        predicate false and the value flows through unchanged; the dead NR
        machinery is swept by the DCE passes that follow. The div-by-zero
        select at the chain's tail (SEL …, 0xffffffff, P1) also still works:
        UREM64 is zero-guarded like SDIV, so it returns instead of trapping
        and the SEL picks the hardware's 0xFFFFFFFF.

        Lifting the chain faithfully instead is WRONG on every backend whose
        MUFU.RCP bits differ from the GPU SFU (the CPU backend's software
        rcp_f32): the NR correction steps are bit-tuned to the hardware
        estimate (B_u64mod case 13, mr-cuda's modular_exponentiation32)."""
        import os, sys, builtins, functools
        dbg = os.environ.get("SLIFTER_INTDIV_DEBUG") == "1"
        print = functools.partial(builtins.print, file=sys.stderr)
        anchor_uses = anchor.get_uses()
        if not anchor_uses:
            return False
        rcp = self._walk_back(anchor, anchor_uses[0], "MUFU", "RCP")
        if rcp is None:
            if dbg: print(f"[u64div] {anchor.id}: no MUFU.RCP")
            return False
        i2f = self._walk_back(rcp, rcp.get_uses()[0], "I2F")
        if i2f is None or "U64" not in i2f.opcodes[1:] \
                or "RP" not in i2f.opcodes[1:]:
            if dbg: print(f"[u64div] {anchor.id}: no I2F.U64.RP ({i2f and i2f.opcodes})")
            return False
        i2f_uses = i2f.get_uses()
        if not i2f_uses or not i2f_uses[0].is_reg_pair:
            if dbg: print(f"[u64div] {anchor.id}: I2F src not pair")
            return False
        divisor_lo_op, divisor_hi_op = i2f_uses[0].reg_group

        f2i = self._find_user(anchor, "F2I")
        if f2i is None or "U64" not in f2i.opcodes[1:] \
                or "TRUNC" not in f2i.opcodes[1:]:
            if dbg: print(f"[u64div] {anchor.id}: no F2I.U64.TRUNC ({f2i and f2i.opcodes})")
            return False

        slice_ids, slice_insts = self._forward_slice(f2i)

        a_matches = []
        for inst in slice_insts:
            dlo = self._match_rem_lo(inst, slice_ids)
            if dlo is not None:
                a_matches.append((inst, dlo))
        if len(a_matches) != 1:
            if dbg:
                print(f"[u64div] {anchor.id}: {len(a_matches)} A matches "
                      f"(slice={len(slice_insts)}): "
                      + " | ".join(str(a.id) for a, _ in a_matches))
            return False
        a_inst, dividend_lo_op = a_matches[0]

        # B must be the SOLE consumer of A's borrow predicate — anything else
        # reading the pre-correction borrow would observe the inexact value.
        carry_op = a_inst.get_defs()[1]
        carry_users = {u for u, _ in a_inst.users.get(carry_op, set())}
        if len(carry_users) != 1:
            if dbg: print(f"[u64div] {anchor.id}: carry users={len(carry_users)}")
            return False
        b_inst = carry_users.pop()
        dividend_hi_op = self._match_rem_hi(b_inst, slice_ids, a_inst)
        if dividend_hi_op is None:
            if dbg: print(f"[u64div] {anchor.id}: B mismatch: {b_inst.opcodes} {b_inst.id}")
            return False

        operand_tail = [dividend_lo_op, dividend_hi_op,
                        divisor_lo_op, divisor_hi_op]
        for old, opcode in ((a_inst, "UREM64"), (b_inst, "UREM64HI")):
            self._splice_div_pseudo(anchor, old, opcode, operand_tail)

        # If this is a DIVIDE (the pre-correction quotient feeds a Granlund-
        # Möller +1/SEL correction → its corrected value is an OUTPUT, as in
        # 45_dual_gemm's coord=index/N), the remainder rewrite alone is not
        # enough: it only SUPPRESSES the correction, locking in the faithful
        # (software-rcp, wrong) Newton-Raphson quotient estimate. Recover the
        # exact quotient too, splicing UDIV64/UDIV64HI over the pre-correction
        # quotient pair so the suppressed correction now selects the exact
        # value. A pure MODULO has no quotient correction → q stays internal →
        # this is skipped (it lifts faithfully but is dead-code-eliminated).
        q_pair = self._find_precorr_quotient_pair(a_inst, slice_ids)
        if q_pair is not None:
            q_lo_def, q_hi_def = q_pair
            for old, opcode in ((q_lo_def, "UDIV64"), (q_hi_def, "UDIV64HI")):
                self._splice_div_pseudo(anchor, old, opcode, operand_tail)
        # The NR refinement, q computation, and A's borrow def all become
        # unused; the following DCE passes sweep them.
        return True

    def _splice_div_pseudo(self, anchor, old, opcode, operand_tail):
        """Replace `old` with a div/rem pseudo-op (UREM64/UREM64HI/
        UDIV64/UDIV64HI) that defines `old`'s first def from the four-word
        [Dlo,Dhi,dlo,dhi] operand tail."""
        blk = old.parent
        new_inst = Instruction(
            id=f"{anchor.id}_{opcode.lower()}_{old.id}",
            opcodes=[opcode],
            operands=[old.get_defs()[0].clone()]
            + [op.clone() for op in operand_tail],
            parentBB=blk,
            pflag=old.pflag,
        )
        new_inst.use_op_start_idx = 1
        blk.instructions[blk.instructions.index(old)] = new_inst

    @classmethod
    def _find_precorr_quotient_pair(cls, a_inst, slice_ids):
        """Locate the pre-correction QUOTIENT pair (q_lo_def, q_hi_def) of a u64
        DIVIDE chain, or None for a pure modulo.

        A's negated operand is `lo(q*d)`, defined by `IMAD.WIDE.U32 prod,
        q_lo, d_lo, …`; its first multiplicand is the pre-correction quotient
        low word q_lo, and that IMAD.WIDE.U32's defining instruction for q_lo is
        the 64-bit refinement's low-word add `IADD3 q_lo, P, …` — which also
        defines a CARRY predicate P. The high word of the refined 64-bit
        quotient is the SOLE consumer of P (`(I)ADD.X q_hi, …, P`), exactly the
        A→B (rem_lo→rem_hi) carry relationship. This carry pairing is robust
        across the two +1-correction encodings ptxas uses (VIADD, no carry pred,
        in 45_dual_gemm; IADD3 with a carry pred in the standalone microbench).

        The chain is a DIVIDE (q is an OUTPUT, not just an internal step toward
        the remainder) iff q_lo feeds the Granlund-Möller +1 correction (an
        IADD3/VIADD/IADD reading q_lo with imm 1) AND a SEL (the q-vs-q+1 join).
        A pure modulo's pre-correction quotient feeds only the q*d producers.

        Returns the (q_lo_def, q_hi_def) instructions, or None on any structural
        mismatch (fall back to the remainder-only rewrite — never guess)."""
        # lo(q*d) = A's negated, non-RZ operand.
        neg = [u for u in a_inst.get_uses()
               if u.is_reg and u.is_negative_reg and not u.is_rz]
        if len(neg) != 1:
            return None
        qd_lo_def = a_inst.reaching_defs.get(neg[0])
        if qd_lo_def is None or not qd_lo_def.opcodes:
            return None
        if qd_lo_def.opcodes[0] != "IMAD" or "WIDE" not in qd_lo_def.opcodes[1:]:
            return None
        qd_uses = qd_lo_def.get_uses()
        if len(qd_uses) < 1:
            return None
        q_lo_op = qd_uses[0]
        q_lo_def = qd_lo_def.reaching_defs.get(q_lo_op)
        if q_lo_def is None or not q_lo_def.get_defs():
            return None
        # q_lo's def must be inside the NR slice (the refinement result), and an
        # IADD3 producing a carry predicate (the 64-bit low-word add).
        if id(q_lo_def) not in slice_ids:
            return None
        q_lo_defs = q_lo_def.get_defs()
        if len(q_lo_defs) < 2 or not q_lo_defs[1].is_predicate_reg \
                or q_lo_defs[1].is_pt:
            return None
        q_lo_realop = q_lo_defs[0]
        q_lo_users = {u for u, _ in q_lo_def.users.get(q_lo_realop, set())}

        # DIVIDE detector: q_lo feeds a +1 correction (imm 1) and a SEL.
        has_plus1, has_sel = False, False
        for u in q_lo_users:
            if not u.opcodes:
                continue
            op0 = u.opcodes[0]
            if op0 == "SEL":
                has_sel = True
            elif op0 in ("IADD3", "VIADD", "IADD"):
                if any(o.is_immediate and o.immediate_value == 1
                       for o in u.get_uses()):
                    has_plus1 = True
        if not (has_plus1 and has_sel):
            return None

        # q_hi_def = the SOLE consumer of q_lo_def's carry predicate (the
        # refined quotient's high word), mirroring the rem_lo→rem_hi carry.
        carry = q_lo_defs[1]
        hi_users = {u for u, _ in q_lo_def.users.get(carry, set())}
        if len(hi_users) != 1:
            return None
        q_hi_def = hi_users.pop()
        if not q_hi_def.get_defs() or id(q_hi_def) not in slice_ids:
            return None
        if q_hi_def is q_lo_def:
            return None
        return q_lo_def, q_hi_def
