"""Resolve register-indirect SASS CALLs via the bank-4 pointer table.

sm_90 reaches external device functions (and some internal dispatch) through
constant bank 4: the CUDA loader relocates `.nv.constant4` so each slot holds
a function address, and the kernel does

    LDC.64 R8, c[0x4][<slot>]      ; (slot may itself come from a MOV imm)
    ...
    CALL.ABS.NOINC R8

The cubin's `.rela.nv.constant4` section names the symbol in every slot
(parsed into `elf.const4_relocs`), so the call target is statically known —
mirroring how ResolveBRX reads BRX jump tables. Without this pass the CALL is
silently dropped during lifting (known_limitations L5; microbench
`test/sm90_synth/printf_call/`).

Two cases are handled:

* UNDEF function symbols (`vprintf`, `__assertfail`) become a `CALLEXT <name>`
  pseudo-op carrying the CUDA device call ABI operands (def R4 = return value;
  uses R5:R4 and R7:R6 = first two 64-bit params) so SSA/def-use version them
  and the backend can emit a host call.

* Symbols *defined* in the cubin (the FlashAttention dispatch helpers / fp
  reciprocal slowpath family): the slot's symbol value is the callee address.
  When it lands inside the current function's own block range — the
  single-section layout where each device subroutine follows the kernel body in
  `.text.<kernel>` — the register-target `CALL.ABS.NOINC Rx` is rewritten to a
  direct `CALL.REL.NOINC <value>` so HandleCall inlines the callee at the site
  (the SASS register arg/return ABI then flows through the shared register
  namespace), and the trailing `BPT.TRAP` poison is removed. Separate-section
  (rdc) callees, whose value is a per-section offset rather than an in-function
  address, are not yet wired and remain register CALLs.
"""

from transform.transform import SaSSTransform
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.operand import Operand


class ResolveIndirectCall(SaSSTransform):
    def apply(self, module):
        print("=== Start of ResolveIndirectCall ===")
        elf = getattr(module, "elf", None)
        relocs = getattr(elf, "const4_relocs", None) or {}
        resolved = 0
        defined = 0
        fanouts = 0
        for func in module.functions:
            if relocs:
                # Resolve the c[0x4] symbol value -> the in-cubin function its
                # slot names. Each device subroutine cuobjdump dumps as its own
                # `Function :` block restarts its SASS offsets at 0, so a defined
                # symbol's value is matched against the *current* function's own
                # block addresses (the single-section / same-function case, where
                # the callee shares the kernel's address space — FlashAttention's
                # dispatch helpers and our const4_devcall microbench).
                addr_set = {b.addr for b in func.blocks}
                for block in func.blocks:
                    r, d = self._process_block(
                        func, block, relocs, addr_set)
                    resolved += r
                    defined += d
            fanouts += self._fanout_rel_calls(func)
        if resolved or defined or fanouts:
            print(f"ResolveIndirectCall: {resolved} extern call(s) resolved, "
                  f"{fanouts} register-relative call(s) fanned out, "
                  f"{defined} defined-symbol call(s) resolved to direct CALL")
        print("=== End of ResolveIndirectCall ===")

    def _process_block(self, func, block, relocs, addr_set):
        resolved = 0
        defined = 0
        new_insts = []
        for idx, inst in enumerate(block.instructions):
            if not (inst.opcodes and inst.opcodes[0] == "CALL"
                    and "ABS" in inst.opcodes):
                new_insts.append(inst)
                continue
            target = inst.operands[0]
            base_reg = None
            if target.is_reg_group and target.reg_group:
                base_reg = target.reg_group[0].reg
            elif target.is_reg:
                base_reg = target.reg
            if not base_reg:
                new_insts.append(inst)
                continue

            slot = self._find_slot(block.instructions, idx, base_reg)
            if slot is None:
                # The sm_90 idiom routinely spans blocks: the
                # `MOV R17,<slot>; LDC.64 R8,c[0x4][R17]` setup sits in a
                # predecessor while BuildCFG's mbarrier TRYWAIT / BSSY-BSYNC
                # convergence scaffolding splits the `CALL.ABS.NOINC R8` into
                # its own block (FlashAttention's mainloop: 3995 of 4140 calls).
                # Fall back to a flat program-order backward walk across blocks.
                slot = self._find_slot_cross_block(func, block, idx, base_reg)
            entry = relocs.get(slot) if slot is not None else None
            if entry is None:
                new_insts.append(inst)
                continue
            name, value, is_func = entry
            if not is_func:
                new_insts.append(inst)
                continue
            if value is not None:
                # Defined in-cubin device function. The const4 slot value is the
                # callee's address; when it names a block of THIS function (the
                # single-section layout — the callee subroutine sits in the same
                # `.text.<kernel>` after the kernel body, e.g. our const4_devcall
                # helper at 0x260, and FlashAttention's dispatch helpers), rewrite
                # the register-target CALL into a direct `CALL.REL.NOINC <value>`.
                # HandleCall then inlines that callee at the site exactly like any
                # other in-function CALL, so the SASS register-based arg/return ABI
                # flows through the shared register namespace (no cross-function
                # boundary is introduced — the same reason HandleCall *inlines*
                # acyclic callees rather than emitting void()() calls).
                if value in addr_set:
                    direct = Instruction(
                        id=inst.id,
                        opcodes=["CALL", "REL", "NOINC"],
                        operands=[Operand.from_immediate(
                            format(value, "04x"), value)],
                        parentBB=block,
                        pflag=inst.pflag.clone() if inst.pflag else None,
                    )
                    direct.inst_content = f"CALL.REL.NOINC {value:#x}"
                    new_insts.append(direct)
                    # The register CALL's fall-through is the loader's BPT.TRAP
                    # poison block (the indirect-call "should never return here"
                    # trap). With the call now resolved + inlined, that path is
                    # the real post-call continuation — strip the leading BPT so
                    # the inlined callee's return reaches the live code after it.
                    self._strip_bpt_poison(block)
                    defined += 1
                    continue
                # Defined symbol whose address is not a block of this function
                # (separate-section / rdc layout): name-based cross-function
                # resolution is not yet wired — leave as-is (counted as a miss).
                new_insts.append(inst)
                continue

            # UNDEF device-runtime function: CUDA device call ABI passes the
            # first two 64-bit params in R4:R5 and R6:R7 and returns in R4.
            callext = Instruction(
                id=f"{inst.id}_ext",
                opcodes=["CALLEXT", name],
                operands=[
                    Operand.from_reg("R4", "R4"),
                    Operand.create_reg_pair(Operand.from_reg("R4", "R4")),
                    Operand.create_reg_pair(Operand.from_reg("R6", "R6")),
                ],
                parentBB=block,
                pflag=inst.pflag.clone() if inst.pflag else None,
            )
            callext.ext_callee = name
            new_insts.append(callext)
            # When the indirect CALL is a block terminator (the FlashAttention
            # mainloop shape), its fall-through successor leads with the loader's
            # BPT.TRAP poison; with the call resolved to a CALLEXT that returns
            # normally, that path is the real continuation, so drop the trap.
            # (No-op for the mid-block printf_call shape, which has no BPT.)
            if inst is block.instructions[-1]:
                self._strip_bpt_poison(block)
            resolved += 1
        block.instructions = new_insts
        return resolved, defined

    def _find_slot_cross_block(self, func, call_block, call_idx, base_reg):
        """Cross-block fallback for `_find_slot`: flatten the function's blocks
        into program order and run the same backward reaching-def walk for the
        `LDC.64 <base_reg>, c[0x4][...]` (and its index-register immediate). The
        sm_90 indirect-call setup is hoisted above the CALL's block by BuildCFG's
        convergence scaffolding, so the `LDC.64`/`MOV <idx>` live in an earlier
        block; the function's flat order preserves the original SASS sequence
        (blocks are appended in address order), so a linear backward scan finds
        them. This mirrors `_collect_umov_candidates`' flat-order walk."""
        flat = [ins for b in func.blocks for ins in b.instructions]
        try:
            flat_idx = flat.index(call_block.instructions[call_idx])
        except ValueError:
            return None
        return self._find_slot(flat, flat_idx, base_reg)

    def _strip_bpt_poison(self, call_block):
        """The sm_90 indirect-device-call idiom places a `BPT.TRAP` immediately
        after `CALL.ABS.NOINC` — the "this indirect call never returns here"
        loader poison (lifted to llvm.trap -> ud2). Once the call is resolved to
        a direct, inlined CALL, that fall-through is the genuine continuation, so
        the trap must go. Remove leading BPT instructions from the call block's
        fall-through successor(s). Scoped to leading BPTs only: a BPT that is not
        at the head of the post-call block is unrelated and left intact."""
        for succ in call_block.succs:
            while (succ.instructions
                   and succ.instructions[0].opcodes
                   and succ.instructions[0].opcodes[0] == "BPT"):
                succ.instructions.pop(0)

    def _fanout_rel_calls(self, func):
        """Rewrite `CALL.REL.NOINC Rx 0x0` whose register is built from
        literal UMOV immediates (predicate-selected section-local function
        addresses — the in-cubin function-pointer dispatch shape, see
        test/sm90_synth/indirect_call/) into a PBRA-guarded pair of direct
        CALL.REL calls, which HandleCall then extracts/inlines as usual.

        MVP scope: exactly one unpredicated default UMOV plus one predicated
        override (the 2-target select). Other shapes are left untouched.
        """
        count = 0
        for block in list(func.blocks):
            for idx, inst in enumerate(block.instructions):
                if not (inst.opcodes and inst.opcodes[0] == "CALL"
                        and "REL" in inst.opcodes):
                    continue
                target = inst.operands[0]
                if not (target.is_reg and target.reg
                        and not target.is_reg_group):
                    continue
                cands = self._collect_umov_candidates(func, block, idx,
                                                      target.reg)
                if cands is None:
                    continue
                pred_op, pred_imm, default_imm = cands
                self._split_for_fanout(
                    func, block, idx, pred_op, pred_imm, default_imm)
                count += 1
                break  # block.instructions changed; outer loop re-walks
        return count

    def _collect_umov_candidates(self, func, call_block, call_idx, reg):
        """Walk back from the CALL in flat program order (across blocks) for
        the UMOV literals selecting the call target. BuildCFG has already
        moved each predicated UMOV into its own PBRA-guarded block (its
        addr_content carries the synthetic "8000" prefix) with the pflag
        stripped — the guard condition is the preceding block's PBRA operand.
        Returns (pred_operand, pred_imm, default_imm) or None."""
        flat = []
        for b in func.blocks:
            for i, ins in enumerate(b.instructions):
                flat.append((b, i, ins))
        try:
            start = flat.index((call_block, call_idx,
                                call_block.instructions[call_idx]))
        except ValueError:
            return None

        def guard_pred(b):
            # PBRA-guarded block: synthetic high-bit address prefix; the
            # guard is the unique pred block's PBRA condition operand.
            if not (b.addr_content or "").startswith("8000"):
                return None
            for p in b.preds:
                for ins in reversed(p.instructions):
                    if ins.opcodes and ins.opcodes[0] == "PBRA":
                        return ins.operands[0]
            return None

        cur = reg
        pred = None
        pred_imm = None
        for j in range(start - 1, -1, -1):
            b, _, inst = flat[j]
            defs = inst.get_defs()
            if not any(self._defines_reg(d, cur) for d in defs):
                continue
            op0 = inst.opcodes[0] if inst.opcodes else ""
            uses = inst.get_uses()
            gp = guard_pred(b)
            if op0 in ("MOV", "UMOV") and uses and uses[0].is_reg \
                    and inst.pflag is None and gp is None:
                cur = uses[0].reg
                continue
            if op0 in ("MOV", "UMOV") and uses and uses[0].is_immediate:
                eff_pred = inst.pflag if inst.pflag is not None else gp
                if eff_pred is not None:
                    if pred is not None:
                        return None  # >2 targets: out of MVP scope
                    pred = eff_pred.clone()
                    pred_imm = int(uses[0].immediate_value)
                    continue
                if pred is None or pred_imm is None:
                    return None  # single target or unexpected order
                return (pred, pred_imm, int(uses[0].immediate_value))
            return None
        return None

    def _split_for_fanout(self, func, block, call_idx, pred_op, pred_imm,
                          default_imm):
        call = block.instructions[call_idx]
        tail = block.instructions[call_idx + 1:]
        block.instructions = block.instructions[:call_idx]

        base = call.id
        new_join = bool(tail) or not block.succs
        if new_join:
            join_bb = BasicBlock(f"{base}_e2", None, tail)
        else:
            # CALL was the block terminator: the existing fall-through
            # successor is the join.
            join_bb = block.succs[0]
        # Predicated target block (taken when pred holds — the predicated
        # UMOV overrode the default).
        a_bb = BasicBlock(f"{base}_e0", None, [])
        b_bb = BasicBlock(f"{base}_e1", None, [])
        for bb, imm in ((a_bb, pred_imm), (b_bb, default_imm)):
            # The CALL must be the block TERMINATOR for HandleCall's
            # collect_call_usage / inlining to engage; the continuation is
            # the succ edge to the join block (HandleCall wires the return).
            direct = Instruction(
                id=f"{bb.addr_content}_call",
                opcodes=["CALL", "REL", "NOINC"],
                operands=[Operand.from_immediate(format(imm, "04x"), imm)],
                parentBB=bb,
            )
            bb.instructions = [direct]
        # NOTE: pred semantics — `@!UPx UMOV` means the override applies when
        # the pflag (possibly negated) holds; PBRA evaluates the same operand
        # form, so pass it through unchanged.
        pbra = Instruction(
            id=f"{base}_pbra",
            opcodes=["PBRA"],
            operands=[
                pred_op,
                Operand.from_immediate(a_bb.addr_content, a_bb.addr),
                Operand.from_immediate(b_bb.addr_content, b_bb.addr),
            ],
            parentBB=block,
        )
        block.instructions.append(pbra)

        # Rewire CFG.
        if new_join:
            join_bb.succs = list(block.succs)
            for s in list(block.succs):
                s.preds = [join_bb if p is block else p for p in s.preds]
            join_bb.preds = [a_bb, b_bb]
            for inst in tail:
                inst.parent = join_bb
        else:
            join_bb.preds = [p for p in join_bb.preds if p is not block]
            join_bb.preds += [a_bb, b_bb]
        block.succs = [a_bb, b_bb]
        for bb in (a_bb, b_bb):
            bb.preds = [block]
            bb.succs = [join_bb]

        pos = func.blocks.index(block)
        insert = [a_bb, b_bb] + ([join_bb] if new_join else [])
        func.blocks[pos + 1:pos + 1] = insert

    def _find_slot(self, insts, call_idx, base_reg):
        """Walk backward from the CALL for `LDC.64 <base_reg>, c[0x4][...]`,
        then (if register-indexed) one more hop for the index immediate."""
        for j in range(call_idx - 1, -1, -1):
            inst = insts[j]
            defs = inst.get_defs()
            if not any(self._defines_reg(d, base_reg) for d in defs):
                continue
            if not (inst.opcodes and inst.opcodes[0] in ("LDC", "ULDC")
                    and "64" in inst.opcodes):
                return None
            src = inst.get_uses()[0]
            if not src.is_const_mem or src.const_mem_bank != 4:
                return None
            if src.is_reg and src.reg and src.reg not in ("RZ", "URZ"):
                idx_imm = self._find_imm_def(insts, j, src.reg)
                if idx_imm is None:
                    return None
                return idx_imm + int(src.offset_value or 0)
            return int(src.offset_value or 0)
        return None

    def _find_imm_def(self, insts, before_idx, reg):
        for j in range(before_idx - 1, -1, -1):
            inst = insts[j]
            if not any(self._defines_reg(d, reg) for d in inst.get_defs()):
                continue
            op0 = inst.opcodes[0] if inst.opcodes else ""
            uses = inst.get_uses()
            # MOV Rx, imm
            if op0 in ("MOV", "UMOV") and uses and uses[0].is_immediate:
                return int(uses[0].immediate_value)
            # IMAD.MOV.U32 Rx, RZ, RZ, imm
            if (op0 == "IMAD" and "MOV" in inst.opcodes and len(uses) >= 3
                    and uses[0].is_rz and uses[1].is_rz
                    and uses[2].is_immediate):
                return int(uses[2].immediate_value)
            if op0 == "SETZERO":
                return 0
            return None
        return None

    @staticmethod
    def _defines_reg(def_op, reg):
        if def_op.is_reg_group and def_op.reg_group:
            return any(s.reg == reg for s in def_op.reg_group)
        return def_op.reg == reg
