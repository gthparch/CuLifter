import re

from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.operand import Operand
from transform.transform import SaSSTransform


_HEX_ID_RE = re.compile(r"^[0-9a-fA-F]+$")


def _is_hex_id(value):
    return bool(_HEX_ID_RE.match(value or ""))


def _nop(inst):
    """Turn an instruction into a harmless NOP."""
    inst.opcodes = ["NOP"]
    inst.operands = []
    inst.inst_content = "NOP"
    inst.use_op_start_idx = 0


class ResolveBRX(SaSSTransform):
    def apply(self, module):
        elf = module.elf
        for func in module.functions:
            self._process_function(func, elf)

    def _process_function(self, func, elf):
        insts = self._sorted_real_insts(func)
        addr_set = {int(inst.id, 16) for inst in insts}
        addr_list = sorted(addr_set)
        max_addr = addr_list[-1]

        def align_up_to_inst(addr):
            while addr not in addr_set and addr <= max_addr:
                addr += 0x8
            return addr

        # ---- Phase 1: collect all BRX info without mutating anything ----
        brx_info = []  # list of (block, brx_inst, brx_pos, selector_op, aligned_targets, nop_positions)
        for block in func.blocks:
            for inst in list(block.instructions):
                if not inst.is_indirect_branch():
                    continue

                brx_pos = block.instructions.index(inst)
                info = self._analyze_brx(block.instructions, brx_pos, elf, func.name)
                if info is None:
                    continue

                selector_op, raw_targets, imnmx_pos, ldc_pos, nop_positions = info
                aligned = [align_up_to_inst(tgt) for tgt in raw_targets]
                _ = aligned[0]
                brx_info.append((block, inst, brx_pos, selector_op,
                                 aligned, imnmx_pos, nop_positions))

        # ---- Phase 2: mutate blocks (reverse order to keep positions valid) ----
        brx_targets = {}
        target_addrs = {}

        for block, brx_inst, brx_pos, selector_op, aligned, imnmx_pos, nop_positions in brx_info:
            seq = block.instructions

            # 2a. Insert synthetic MOV right after IMNMX to preserve selector.
            mov_inst = self._make_selector_mov(seq[imnmx_pos], selector_op)
            seq.insert(imnmx_pos + 1, mov_inst)

            # Adjust all positions after the insertion.
            brx_pos += 1
            shifted_nop = [p + 1 if p > imnmx_pos else p for p in nop_positions]

            # 2b. NOP the BRX address-computation chain.
            for pos in shifted_nop:
                _nop(seq[pos])

            # 2c. Replace BRX with SWITCH.
            switch_inst = Instruction(
                id=brx_inst.id,
                opcodes=["SWITCH"],
                operands=[selector_op],
                parentBB=block,
                pflag=brx_inst.pflag,
            )
            switch_inst.inst_content = f"SWITCH {selector_op}"
            switch_inst.ctl_code = brx_inst.ctl_code
            seq[brx_pos] = switch_inst

            brx_targets[switch_inst] = aligned
            for tgt in aligned:
                target_addrs[tgt] = None

        # ---- Phase 3: split blocks at target addresses and wire up CFG ----
        for addr in target_addrs.keys():
            self._ensure_block_boundary(func, f"{addr:04x}")

        block_by_addr = {bb.addr: bb for bb in func.blocks}
        for switch_inst, targets in brx_targets.items():
            brx_block = switch_inst.parent
            target_blocks = [block_by_addr[tgt] for tgt in targets]
            # Store ordered target list on SWITCH instruction for the lifter.
            # This preserves duplicate targets (e.g., default == case N) that
            # get collapsed in the deduplicated succs/preds lists.
            switch_inst.switch_targets = target_blocks
            for succ in target_blocks:
                brx_block.add_succ(succ)
                succ.add_pred(brx_block)

    # ------------------------------------------------------------------
    # Analysis (read-only — does NOT mutate the instruction list)
    # ------------------------------------------------------------------

    def _analyze_brx(self, seq, brx_pos, elf, kernel):
        """Return (selector_op, raw_targets, imnmx_pos, ldc_pos, nop_positions)
        or None if the pattern cannot be matched."""
        # Find the LDC that loads from constant bank (jump table).
        ldc_pos = None
        for pos in range(brx_pos - 1, -1, -1):
            if seq[pos].opcodes[0] == "LDC":
                ldc_pos = pos
                break
        if ldc_pos is None:
            return None

        ldc_inst = seq[ldc_pos]
        const_op = ldc_inst.get_uses()[0]

        # Find the IMNMX/VIMNMX clamp before the LDC.
        imnmx_pos = None
        for pos in range(ldc_pos - 1, -1, -1):
            if seq[pos].opcodes[0] in ("IMNMX", "VIADDMNMX", "VIMNMX"):
                imnmx_pos = pos
                break
        if imnmx_pos is None:
            return None

        imnmx_inst = seq[imnmx_pos]
        bound_idx = 2 if imnmx_inst.opcodes[0] == "VIADDMNMX" else 1
        bound_op = imnmx_inst.get_uses()[bound_idx]
        # The jump table is only statically resolvable when the clamp bounds the
        # selector with a compile-time LITERAL upper bound. Some kernels (e.g.
        # deredundancy, logic-rewrite) clamp against a *register* (non-constant
        # bound), so this BRX is not a constant-table switch we can lower.
        # Decline (caller leaves the BRX unresolved) rather than crash int(None).
        if bound_op.immediate_value is None:
            return None
        ub = int(bound_op.immediate_value)

        # Build selector_op — a fresh register for the synthetic MOV.
        sel_reg = f"RBRX_sel_{imnmx_inst.id}"
        selector_op = Operand.from_reg(sel_reg, sel_reg)

        # Identify positions to NOP: stride IMAD, LDC, SHF (sign-extend).
        nop_positions = [ldc_pos]

        # The IMAD that computes the byte offset (index * 4) between IMNMX
        # and LDC.
        for pos in range(imnmx_pos + 1, ldc_pos):
            if seq[pos].opcodes[0] == "IMAD":
                nop_positions.append(pos)
                break

        # The SHF sign-extension between LDC and BRX.
        for pos in range(ldc_pos + 1, brx_pos):
            if seq[pos].opcodes[0] == "SHF":
                nop_positions.append(pos)
                break

        # Read jump table entries from constant bank.
        bank = int(const_op.const_mem_bank)
        base_offset = int(const_op.offset_value)
        bank_size = elf.const_size_bytes(kernel, bank)
        max_entries = (bank_size - base_offset) // 4
        count = min(ub + 1, max_entries)
        targets = [
            int(elf.read_const_u32(kernel, bank, base_offset + idx * 4))
            for idx in range(count)
        ]
        if not targets:
            targets = [0] * (ub + 1)

        return selector_op, targets, imnmx_pos, ldc_pos, nop_positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_selector_mov(self, imnmx_inst, sel_use_op):
        """Create a synthetic MOV that copies the IMNMX/VIMNMX output to a
        fresh register so the SWITCH selector survives any later redefinition
        of the original register."""
        imnmx_def = imnmx_inst.get_defs()[0]
        sel_reg = sel_use_op.reg
        sel_def_op = Operand.from_reg(sel_reg, sel_reg)
        mov_use_op = imnmx_def.clone()
        mov_inst = Instruction(
            id=f"{imnmx_inst.id}_brx_mov",
            opcodes=["MOV"],
            operands=[sel_def_op, mov_use_op],
            parentBB=imnmx_inst.parent,
            pflag=None,
        )
        mov_inst.inst_content = f"MOV {sel_reg}, {imnmx_def.name}"
        mov_inst.use_op_start_idx = 1
        return mov_inst

    def _sorted_real_insts(self, func):
        insts = []
        for bb in func.blocks:
            for inst in bb.instructions:
                if _is_hex_id(inst.id):
                    insts.append(inst)
        insts.sort(key=lambda inst: int(inst.id, 16))
        return insts

    def _ensure_block_boundary(self, func, inst_id):
        inst_loc = {}
        for bb in func.blocks:
            for idx, inst in enumerate(bb.instructions):
                inst_loc[inst.id.lower()] = (bb, idx)

        key = f"{int(inst_id, 16):04x}"
        block, idx = inst_loc[key]
        if idx == 0:
            return block

        return self._split_block(func, block, idx)

    def _split_block(self, func, block, split_idx):
        head_insts = list(block.instructions[:split_idx])
        tail_insts = list(block.instructions[split_idx:])
        tail = BasicBlock(tail_insts[0].id, block.pflag, tail_insts)

        old_succs = list(block.succs)
        block.succs = []
        tail.succs = old_succs

        for succ in old_succs:
            succ.preds.remove(block)
            succ.add_pred(tail)

        tail.preds = []
        tail.add_pred(block)
        block.add_succ(tail)

        block.instructions = head_insts
        for inst in block.instructions:
            inst.parent = block
        for inst in tail.instructions:
            inst.parent = tail

        dest_op = Operand.from_immediate(tail.addr_content, int(tail.addr_content, 16))
        bra_inst = Instruction(
            id=f"{block.addr_content}_syn_split",
            opcodes=["BRA"],
            operands=[dest_op],
            parentBB=block,
            pflag=None,
        )
        block.instructions.append(bra_inst)

        blocks = func.blocks
        pos = blocks.index(block)
        blocks.insert(pos + 1, tail)
        func.blocks = blocks

        return tail
