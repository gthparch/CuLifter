from transform.transform import SaSSTransform


class RecordWgmmaSwizzle(SaSSTransform):
    """Record the per-operand GMMA swizzle mode for each descriptor-form
    HGMMA/WGMMA/QGMMA, BEFORE SSA/RegRemap rename the descriptor build chain.

    A wgmma.mma_async takes two 64-bit matrix descriptors laid out as the
    consecutive uniform-register pairs [UR(n)=A.lo, UR(n+1)=A.hi, UR(n+2)=B.lo,
    UR(n+3)=B.hi] for a GEMM spelled `gdesc[URn]`. Each descriptor's HIGH word
    carries the cute GMMA::LayoutType (swizzle mode) in bits[63:62] == high-word
    bits[31:30]:
        lt = 1 -> SW128 (atom 64 elems, swizzle bit-width B=3)
        lt = 2 -> SW64  (atom 32 elems, B=2)
        lt = 3 -> SW32  (atom 16 elems, B=1)
        lt = 0 -> none/K-INTER (atom 8 elems, B=0)
    i.e. B = 4 - lt for lt in {1,2,3}, else 0. Verified vs the FA3 hd96 bwd SASS
    (A=K desc 0xc0000010 = lt3 = SW32 B1; B=Q/dO desc 0x80000020 = lt2 = SW64 B2)
    and CUTE's SmemLayout atoms.

    Most FA3 kernels (hd128/192/256 + all fwd) are uniformly SW128, but hd96
    MIXES per-operand swizzle, so the lifter must read each operand's mode
    separately. The HIGH words are built by `UMOV URx, <imm>` directly before the
    GEMM; the SAME UR is REUSED across GEMMs with DIFFERENT layouts (e.g. UR13
    holds 0xc0000010 for the QK GEMM then 0x40000040 for the dKV GEMM), so a
    function-wide map is ambiguous — this pass resolves PER-GEMM via a linear
    BACKWARD scan (most-recent layout write before each GEMM in block order).
    It stashes `inst._swiz_B_a` / `inst._swiz_B_b` on each GEMM. It emits NO IR
    and does NOT touch the operand/def-use graph, so SW128 kernels stay byte-
    identical (the lifter defaults both to B=3 when unrecorded, and B=3 reproduces
    the original fixed SW128 swizzle exactly).
    """

    _GMMA_OPS = ("HGMMA", "WGMMA", "QGMMA")
    _MOV_OPS = ("MOV", "UMOV", "IADD3", "UIADD3")

    @staticmethod
    def _layout_B_from_imm(iv):
        """Decode a GMMA matrix-descriptor HIGH word -> swizzle bit-width B, or
        None if `iv` is not a valid descriptor high word.

        The 64-bit descriptor HIGH word (cute mma_sm90_desc.hpp) is:
          bits[0,14)  = stride_byte_offset
          bits[14,17) = unused (0)
          bits[17,20) = base_offset (3 bits)
          bits[20,30) = unused (0)        <-- the discriminator
          bits[30,32) = layout_type       (INTERLEAVE=0,B128=1,B64=2,B32=3)
        So B = 4 - layout_type for layout_type in {1,2,3}, else swizzle-less.

        Many SASS UMOVs near a GEMM load unrelated stride/mask words (e.g.
        0x7ffff800, 0x7ff80000) whose bit30 happens to be set; those are NOT
        descriptor high words. Require the unused gap bits[20,30) == 0 (the real
        swizzle immediates 0x40000040/0x80000020/0xc0000010 all have it clear) to
        reject them — else a K-INTER HGMMA (e.g. test/sm90_synth/tma_hgmma) would
        be misrouted to the swizzle path."""
        if not isinstance(iv, int):
            return None
        if (iv >> 20) & 0x3FF:        # bits[20,30) must be 0 for a real desc word
            return None
        lt = (iv >> 30) & 0x3
        if lt == 0:
            return None          # INTERLEAVE/no-swizzle (or a zero/carry operand)
        return 4 - lt            # lt 1/2/3 (B128/B64/B32) -> B 3/2/1

    @classmethod
    def _layout_write(cls, inst):
        """If `inst` is a (U)MOV/(U)IADD3 writing a layout immediate, return
        (dst_reg_name, B); else (None, None)."""
        if not inst.opcodes or inst.opcodes[0] not in cls._MOV_OPS:
            return None, None
        defs = inst.get_defs()
        if not defs:
            return None, None
        dst = defs[0]
        name = getattr(dst, "reg", None)
        if name is None and getattr(dst, "is_reg_group", False) and dst.reg_group:
            name = dst.reg_group[0].reg
        if name is None:
            return None, None
        for u in inst.get_uses():
            if getattr(u, "is_immediate", False):
                B = cls._layout_B_from_imm(getattr(u, "immediate_value", None))
                if B is not None:
                    return name, B
        return None, None

    @classmethod
    def _copy_write(cls, inst):
        """If `inst` is a plain `(U)MOV URdst, URsrc` uniform-register copy,
        return (dst_reg_name, src_reg_name); else (None, None).

        A software-pipelined mainloop rebuilds a GEMM's descriptor quad through
        register COPIES rather than reloading the layout immediate (FA3 hd96 fwd
        steady-state QK: `UMOV UR37, UR13` copies the SW64 high word set up in the
        prologue). The backward scan must follow such copies to the immediate, or
        the GEMM's swizzle mode is silently unrecorded (defaults to SW128) at that
        site only — while the prologue site records correctly. That per-site split
        corrupts every iteration after the first."""
        if not inst.opcodes or inst.opcodes[0] not in ("MOV", "UMOV"):
            return None, None
        defs = inst.get_defs()
        if not defs:
            return None, None
        dst = getattr(defs[0], "reg", None)
        if dst is None:
            return None, None
        uses = inst.get_uses()
        if len(uses) != 1:
            return None, None
        src = getattr(uses[0], "reg", None)
        if src is None or not src.startswith("UR"):
            return None, None
        return dst, src

    def apply(self, module):
        super().apply(module)
        for func in module.functions:
            # Flatten to block order; for each GEMM, scan backward for the most
            # recent layout-immediate write to A.hi = UR(n+1) and B.hi = UR(n+3).
            flat = []
            for block in func.blocks:
                for inst in block.instructions:
                    flat.append(inst)
            for idx, inst in enumerate(flat):
                if not inst.opcodes or inst.opcodes[0] not in self._GMMA_OPS:
                    continue
                desc_reg = self._find_desc_reg(inst)
                if desc_reg is None or not desc_reg.startswith("UR"):
                    continue
                try:
                    n = int(desc_reg[2:])
                except ValueError:
                    continue
                want_a = f"UR{n + 1}"
                want_b = f"UR{n + 3}"
                B_a = B_b = None
                # backward scan (bounded) for the latest layout write to each.
                # Register copies to the tracked high word retarget the scan to
                # the copy source (see _copy_write) so pipelined descriptor
                # rebuilds resolve to the prologue's layout immediate.
                # Scan bound: 4000 instructions covers the largest observed
                # prologue->mainloop distance (FA3) with wide margin while keeping
                # the pass O(GEMMs * bound). A GEMM whose layout write sits farther
                # back is left unrecorded and the lifter defaults it to SW128
                # (B=3) — the same behavior as before this pass existed.
                for j in range(idx - 1, max(idx - 4000, -1), -1):
                    if B_a is not None and B_b is not None:
                        break
                    wname, wB = self._layout_write(flat[j])
                    if wname is not None:
                        if wname == want_a and B_a is None:
                            B_a = wB
                        if wname == want_b and B_b is None:
                            B_b = wB
                        continue
                    cdst, csrc = self._copy_write(flat[j])
                    if cdst is None:
                        continue
                    if cdst == want_a and B_a is None:
                        want_a = csrc
                    if cdst == want_b and B_b is None:
                        want_b = csrc
                if B_a is not None:
                    inst._swiz_B_a = B_a
                if B_b is not None:
                    inst._swiz_B_b = B_b

    @staticmethod
    def _find_desc_reg(inst):
        """The gdesc[URn] base register name (original parse-time name)."""
        for op in inst.operands:
            if getattr(op, "desc_hi", False) or getattr(op, "desc_a_hi", False):
                continue
            dr = getattr(op, "desc_reg", None)
            if dr and dr.startswith("UR"):
                return dr
        return None
