import re

from sir.operand import Operand
from sir.controlcode import ControlCode
from sir.controlcode import PresetCtlCodeException

class Instruction:
    # Backend gate: --lifter nvvm sets this True (main.py) so the parser
    # synthesizes desc_b's HIGH word for full wgmma descriptors; the x86warp
    # path keeps it False to preserve its verified SSA numbering.
    SYNTH_DESC_B_HI = False

    def __init__(self, id, opcodes, operands, parentBB=None, pflag=None, encoding=None):
        self.id = id
        self.opcodes = opcodes
        self.operands = operands
        self.inst_content = None
        self.true_branch = None
        self.false_branch = None
        self.ctl_code = None
        self.parent = parentBB
        self.pflag = pflag
        self.encoding = encoding

        self.users = {}
        self.reaching_defs = {}

        # Def/use operands layout correction
        self.use_op_start_idx = 1
        
        
        # ------ determine use operand start index ------
        
        if len(self.opcodes) > 0 and (self.opcodes[0] == "PHI" or self.opcodes[0] == "PHI64"):
            return            
        
        elif len(self.opcodes) > 1 and self.opcodes[0] == "HMMA":
            shape = self.opcodes[1] if len(self.opcodes) > 1 else ""
            accum = self.opcodes[2] if len(self.opcodes) > 2 else ""
            if shape == "884":
                self.use_op_start_idx = 2
            elif shape == "1688" and accum == "F16":
                self.use_op_start_idx = 2
            elif shape == "16816" and accum == "F16":
                self.use_op_start_idx = 2
            else:
                self.use_op_start_idx = 4

        elif self.opcodes == ["FFMA2"] and (len(self.operands) == 8 or len(self.operands) == 4 and all(
                op.is_reg and op.prefix in (None, "", "-") and (op.suffix or "").upper() in
                ("F32", "F32X2.HI_LO", "F32X2.HI_LO.NP", "F32X2.LO_HI", "F32X2.LO_HI.NP")
                for op in self.operands[1:])):
            self.use_op_start_idx = 2

        elif self.opcodes[:3] == ["QMMA", "16832", "F32"]:
            self.use_op_start_idx = 4

        # HGMMA/WGMMA/QGMMA: warpgroup MMA (SM90)
        # HGMMA.64x8x16.F32 D, A, B, C  (register form, 4 operands)
        # HGMMA.64x128x16.F32 D, gdesc[UR8].tnspA.tnspB, C  (descriptor form, 3+ operands)
        # QGMMA.64x224x32.F32.E4M3.E4M3 — FP8 (e4m3/e5m2) warpgroup MMA; same
        # descriptor/register operand layout as HGMMA (D count = N/2 for F32),
        # only the input element type (1-byte fp8) and K-step (32) differ.
        elif len(self.opcodes) > 1 and self.opcodes[0] in ("HGMMA", "WGMMA", "QGMMA"):
            shape_str = self.opcodes[1]  # e.g. "64x8x16"
            accum = self.opcodes[2] if len(self.opcodes) > 2 else "F32"
            d_count = self._wgmma_d_reg_count(shape_str, accum)
            self.use_op_start_idx = d_count

        elif len(self.opcodes) > 1 and self.opcodes[0] == "IMMA":
            # Sparse IMMA (IMMA.SP.<shape>.S8.S8[.SAT]) has 4 defs (m/n tile).
            shape = self.opcodes[2] if self.opcodes[1] == "SP" else self.opcodes[1]
            self.use_op_start_idx = 2 if shape in ("8816", "8832") else 4

        # DMMA.<shape> D, A, B, C — F64 tensor-core MMA.
        # Per-thread D f64 count (= def pair-operands after expansion):
        # m8n8k4 → 2; m16n8k8/m16n8k16 → 4.
        elif len(self.opcodes) > 1 and self.opcodes[0] == "DMMA":
            self.use_op_start_idx = 2 if self.opcodes[1] == "8x8x4" else 4

        # DSETP.LT.AND P0, PT, R4, R6, PT ; R5:R4, R7:R6 are 64-bit pairs
        elif self.opcodes[0] == "DSETP":
            self.use_op_start_idx = 2

        # CALL.ABS.NOINC R14 ; R15:R14 is 64-bit call target
        elif self.opcodes[0] in ("CALL", "CALLFUNC"):
            self.use_op_start_idx = 0

        # ATOM.E.ADD.F16x2.RN.STRONG.GPU P0, RZ, [R2], R5 ;
        # P0 = predicate def (success), RZ = old value def, [R2] = mem addr, R5 = src
        elif len(self.opcodes) > 1 and self.opcodes[0] == "ATOM" and "E" in self.opcodes:
            self.use_op_start_idx = 2

        # ATOMG.E.CAS.STRONG.GPU PT, R5, [R4], R2, R3 ;
        elif len(self.opcodes) > 1 and self.opcodes[0] == "ATOMG" and "E" in self.opcodes:
            self.use_op_start_idx = 2

        # BAR.SYNC 0x0
        elif len(self.opcodes) > 1 and self.opcodes[0] == "BAR":
            self.use_op_start_idx = 0
            
        # WAROPSYNC 0x0
        elif self.opcodes[0] == "WARPSYNC":
            self.use_op_start_idx = 0

        # VOTE.ALL UP3, PT, P5 or VOTEU.ALL UP3, P5
        elif self.opcodes[0] in ("VOTE", "VOTEU"):
            self.use_op_start_idx = 1

        # ELECT P0, URZ, PT — two defs (predicate, uniform register), one use (guard predicate)
        elif self.opcodes[0] == "ELECT":
            self.use_op_start_idx = 2

        # B2R.RESULT Rdst, Pdst - both are defs
        elif self.opcodes[0] == "B2R":
            self.use_op_start_idx = 2

        # SHFL.DOWN PT, R59 = R18, 0x8, 0x1f
        elif len(self.opcodes) > 1 and self.opcodes[0] == "SHFL":
            self.use_op_start_idx = 2

        # ISETP.LT.U32.OR = P0, PT, R12, R10, P0;
        elif "SETP" in self.opcodes[0]:
            self.use_op_start_idx = 2

        # LOP3.LUT R2 = R2, 0xff, RZ, 0xc0, !PT ;
        # PLOP3.LUT P1, PT = PT, PT, PT, 0x8, 0x0 ; 
        # LOP3.LUT P0, RZ = R3, 0x1f, RZ, 0xc0, !PT ;
        elif len(self.opcodes) > 1 and "LOP3" in self.opcodes[0] and self.opcodes[1] == "LUT":
            if len(self.operands) > 0 and (self.operands[0].is_predicate_reg or self.operands[0].is_pt):
                self.use_op_start_idx = 2
            else:
                self.use_op_start_idx = 1

        # ST.E.SYS [R2], R5 ;
        # STG.E.64.SYS [R2+0x70f800], R4 ;
        # STL.64 [R52+0x110] = R12 uses R12 and R13
        # STS.64 [R66], R168 stores 64-bit from R168:R169
        elif self.is_store():
            self.use_op_start_idx = 0

        # LD.E.SYS R0, [R2] ;
        # LDG.E.64.SYS R4 = [R2] defines R4 and R5
        # LDL.64 R12 = [R52+0x110] defines R12 and R13
        # LDS.U.64 R181 = [R66] loads 64-bit into R181:R182
        elif self.is_load():
            self.use_op_start_idx = 1

        # FIND_AND_SET writes the success predicate and the group index.
        elif self.opcodes[:2] == ["UTCATOMSWS", "FIND_AND_SET"]:
            self.use_op_start_idx = 2

        # Store and Branch have no def op (UBLKCP: TMA bulk copy writes
        # only memory — its dst-address chain must be traced as a USE so
        # DCE keeps it live, same as RED/REDG)
        #
        # UTMALDG/UTMASTG (TMA tensor bulk copy) likewise define no register:
        # the leading `[URdst]` operand is the *shared-memory destination
        # ADDRESS*, i.e. a USE of URdst's value — not a register def. If it is
        # mis-classified as a def, SSA hands it a fresh version that is
        # disconnected from the UPRMT/UIADD chain that actually computed the
        # smem base, so the lifter can never recover the destination offset
        # (it resolves to 0). Treating every operand as a use keeps that link
        # intact and lets DCE keep the address-build chain live.
        # EXIT has NO def — a `@pred EXIT Pg` guard predicate (operands[0]) is
        # a USE, not a def. The default idx=1 mis-typed Pg as a def, so DCE
        # eliminated the SETP that produced it (sheath: the x>=0 bounds
        # predicate guarding `@!P0 EXIT P1` was dropped -> x<0 particles never
        # marked dead -> next scatter OOB).
        # ARRIVES (cp.async mbarrier arrive) likewise defines nothing: its
        # sole `[URb+off]` operand is the mbarrier ADDRESS (a use). Mis-typing
        # it as a def let DCE delete the UMOV that built the barrier base ->
        # KeyError at lift (FA fwd: UR13's def dropped, SYNCS on the same
        # barrier then read an undefined register).
        elif self.is_branch() or self.opcodes[0] in (
                "RED", "REDG", "UBLKCP", "UBLKRED", "UTMALDG", "UTMASTG", "UTCIMMA", "UTCBAR",
                "EXIT"):
            self.use_op_start_idx = 0

        # ARRIVES.LDGSTSBAR.64.TRANSCNT [URdst+off] — mbarrier arrive-on-cp.async.
        # Same shape as UTMASTG/RED: it writes only memory (the mbarrier object at
        # [URdst+off]) and defines NO register. The leading [URdst] is the
        # destination ADDRESS, i.e. a USE of URdst's value. If mis-classified as a
        # def, SSA hands the address-base register a fresh, disconnected version
        # that nothing defines (resolves to 0), so a loop-carried shared-mem base
        # is lost on iterations 2+ — the FA3 packgqa producer's mbarrier arrives
        # then target address 0 and the consumer waits forever (deadlock).
        # Guarded on is_mem_addr so a token-returning form with a leading register
        # def (if one ever appears) keeps the default def layout.
        elif (self.opcodes[0] == "ARRIVES" and self.operands
              and self.operands[0].is_mem_addr):
            self.use_op_start_idx = 0

        elif self.opcodes[0] == "UNPACK64":
            self.use_op_start_idx = 2
            
        else:
            # heuristic: predicate regs at beginning are def operands
            while (self.use_op_start_idx < len(self.operands) and 
                   (self.operands[self.use_op_start_idx].is_predicate_reg)):
                self.use_op_start_idx += 1

        # Ensure consistent operand layout for half instructions
        # E.g. HSETP2.GEU.AND P1, PT, R4.H0_H0, 8.5, 2.0, PT ;
        # => HSETP2.GEU.AND P1, PT, R4.H0_H0, packhalf2(8.5, 2.0), PT ;
        if self.opcodes and self.opcodes[0] == "HSETP2":
            uses = self.operands[self.use_op_start_idx :]
            merged_uses = []
            idx = 0
            while idx < len(uses):
                op = uses[idx]
                if op.is_immediate and op.is_float_immediate:
                    if idx + 1 >= len(uses) or not uses[idx + 1].is_immediate:
                        raise ValueError(f"not possible: immediate comes in pairs")
                    merged_uses.append(
                        Operand.from_half2_immediates(
                            op.immediate_value,
                            uses[idx + 1].immediate_value,
                        )
                    )
                    idx += 2
                    continue

                merged_uses.append(op)
                idx += 1

            self.operands = self.operands[: self.use_op_start_idx] + merged_uses
            
        # ------ restore implicit registers ------

        # NOTE: sm70+ IMAD.HI Rd, Ra, Rb, Rc = hi32(Ra*Rb + {Rc+1:Rc}) — the
        # register addend names a 64-bit PAIR added to the full 64-bit product
        # BEFORE the high-word extraction (same datapath as IMAD.WIDE, which
        # just keeps both result words).  Adding Rc as a 32-bit value after
        # extraction loses the addend's low word and the carry into the high
        # word (broke mont_prod32 in hecbench mr-cuda: u = (t + m*n) >> 32 is
        # a single IMAD.HI.U32 with {R13:R12} = t).
        # When Rd == Ra, Ra is read before Rd is written — no implicit accumulation.
        # (Previously a bug here added dest as an implicit 4th use operand.)

        # IMAD.WIDE R14, P1, R21.reuse, 0x4, R10 ;
        # UIMAD.WIDE.U32 UR6, UR8, UR9, UR6 ;
        if len(self.opcodes) > 1 and (self.opcodes[0] == "IMAD" or self.opcodes[0] == "UIMAD") and self.opcodes[1] == "WIDE":
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            self.operands[2 + self.use_op_start_idx] = Operand.create_reg_pair(self.operands[2 + self.use_op_start_idx])

        # IMAD.HI.U32 R17, R8, R17, R12 ;  addend is pair {R13:R12}
        # Gate on opcode position 1 ("IMAD.HI[.U32]") so Kepler's
        # IMAD.U32.U32.HI(.X) — a true 32-bit addend with .CC/.X carry
        # chains — is excluded.  create_reg_pair maps RZ to {RZ:RZ}=0 and
        # passes immediates/const-mem through unchanged, so this is safe
        # to apply unconditionally to the third use operand.
        elif len(self.opcodes) > 1 and (self.opcodes[0] == "IMAD" or self.opcodes[0] == "UIMAD") and self.opcodes[1] == "HI":
            self.operands[2 + self.use_op_start_idx] = Operand.create_reg_pair(self.operands[2 + self.use_op_start_idx])

        # Blackwell uses the base register of a pair for IADD.64 and MOV.64.
        elif self.opcodes in (["IADD", "64"], ["MOV", "64"]):
            self.opcodes = [self.opcodes[0] + "64"]
            self.operands = [Operand.create_reg_pair(op) for op in self.operands]

        # QSPC.E.S P0, RZ, [R8] ; => R8:R9
        elif len(self.opcodes) > 1 and self.opcodes[0] == "QSPC" and self.opcodes[1] == "E":
            self.operands[2] = Operand.create_reg_pair(self.operands[2])

        # ATOMS.CAST.SPIN.64 R10, [0x820], R8, R10 ;
        # R9:R8 and R11:R10 are the 64-bit register pairs.
        # The destination is ALSO 64-bit: for ADD/EXCH/etc. it holds the old
        # 64-bit value, and for CAST/CAS it holds the 64-bit success result
        # {success, 0}. ptxas's CAS-spin idiom tests the FULL 64-bit result
        # ({R7:R6} == 1 via ISETP + ISETP.EX over R7), so the high word (R7)
        # MUST be written (to 0) — if the dest stays a single 32-bit reg, R7
        # keeps the stale high word of the DADD'd new value and the .EX test
        # (R7 == 0) never passes -> the spin loop deadlocks (aop/atomicPerf/
        # expdist FP64 shared atomicAdd). Pairing the dest makes _set_val split
        # the 64-bit result across R6:R7 and zero R7.
        elif len(self.opcodes) > 3 and self.opcodes[0] == "ATOMS" and "64" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            self.operands[2] = Operand.create_reg_pair(self.operands[2])
            self.operands[3] = Operand.create_reg_pair(self.operands[3])
            
        # BRX R10 -0x1f90 ;
        # R11:R10 is 64-bit target address
        elif self.opcodes[0] == "BRX":
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            
        # CALL.ABS.NOINC R2 ; 
        # R3:R2 is 64-bit call target
        elif self.opcodes[0] == "CALL" and "ABS" in self.opcodes and self.operands[0].is_reg:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # CALL.REL.NOINC R6 0x0 ; register-relative call — cuobjdump separates
        # the register and the relative immediate with a SPACE (no comma), so
        # the splitter mashes them into one token ("R60x0"). Re-split so
        # ResolveIndirectCall sees a register target.
        elif (self.opcodes[0] == "CALL" and "REL" in self.opcodes
              and len(self.operands) == 1 and self.operands[0].name):
            import re as _re
            m = _re.match(r"^(U?R\d+)(-?0x[0-9a-fA-F]+)$",
                          self.operands[0].name)
            if m:
                self.operands = [
                    Operand.from_reg(m.group(1), m.group(1)),
                    Operand.from_immediate(m.group(2), int(m.group(2), 16)),
                ]
        
        # CS2R R26, SRZ ; un-suffixed CS2R is the 64-bit form (CS2R.32 is the
        # 32-bit one) and writes the PAIR R27:R26. Without the pair def, SSA
        # only versions the low register and the high half keeps a stale value
        # from a previous loop iteration (lci-cuda Sum_Omega: alpha(-1) read
        # as 1.0 via leftover 0x3ff00000 in the high word).
        elif self.opcodes[0] == "CS2R" and "32" not in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # DADD/DMUL R4, R2, R6 ; R5:R4, R3:R2, R7:R6 are 64-bit pairs
        elif self.opcodes[0] in ("DADD", "DMUL"):
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            self.operands[1] = Operand.create_reg_pair(self.operands[1])
            self.operands[2] = Operand.create_reg_pair(self.operands[2])
        
        # DFMA R4, R2, R6, R8 ; R5:R4, R3:R2, R7:R6, R9:R8 are 64-bit pairs
        elif self.opcodes[0] == "DFMA":
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            self.operands[1] = Operand.create_reg_pair(self.operands[1])
            self.operands[2] = Operand.create_reg_pair(self.operands[2])
            self.operands[3] = Operand.create_reg_pair(self.operands[3])
        
        # DSETP.LT.AND P0, PT, R4, R6, PT ; R5:R4, R7:R6 are 64-bit pairs
        elif self.opcodes[0] == "DSETP":
            self.operands[2] = Operand.create_reg_pair(self.operands[2])
            self.operands[3] = Operand.create_reg_pair(self.operands[3])

        # FRND.F64.TRUNC R4, R6 ; R5:R4 dest and R7:R6 src are 64-bit pairs.
        # (Float FRND is 32-bit and needs no expansion.) Without this the lifter
        # reads only the low word of the f64 operand and the rounding collapses.
        elif self.opcodes[0] in ("FRND", "UFRND") and "F64" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # P2R Rd, PR, Rs, mask — packs physical predicates P0-P6 (selected by the
        # immediate mask) into Rd. The predicate file (`PR`) is invisible to SSA,
        # so append the masked predicates as explicit USE operands; then SSA/
        # def-use version them and link each to its reaching def (even cross-BB or
        # loop-carried), and lift_p2r reads the correct value. Without this the
        # lifter can't resolve a saved predicate and emits 0 — e.g. nvtiff LZW's
        # `P2R Rd, PR, RZ, 0x20` saving the offset-16 cross-warp scan validity.
        elif self.opcodes[0] == "P2R" and len(self.operands) >= 4 \
                and getattr(self.operands[3], "is_immediate", False) \
                and isinstance(self.operands[3].immediate_value, int):
            mask = self.operands[3].immediate_value
            for bit in range(7):
                if mask & (1 << bit):
                    self.operands.append(Operand.from_reg(f"P{bit}", f"P{bit}"))

        # UP2UR URd, UPR, URs, mask — the uniform-datapath analog of P2R:
        # packs the masked UNIFORM predicates (UP0..UP6) into URd. Same SSA
        # trick: append the masked UP<i> as explicit uses so def-use links
        # each to its reaching UISETP def. Without this the `UPR` placeholder
        # never resolves and the packed value lifts to garbage — FA3 bwd
        # hdim256 reads `UP2UR UR12, UPR, URZ, 0x1` for its mbarrier-phase
        # bookkeeping and produced dQ=inf.
        elif self.opcodes[0] == "UP2UR" and len(self.operands) >= 4 \
                and getattr(self.operands[3], "is_immediate", False) \
                and isinstance(self.operands[3].immediate_value, int):
            mask = self.operands[3].immediate_value
            for bit in range(7):
                if mask & (1 << bit):
                    self.operands.append(Operand.from_reg(f"UP{bit}", f"UP{bit}"))

        # R2P PR, Rs, mask — the inverse: writes the masked physical predicates
        # from Rs's bits. Make them explicit DEF operands (replacing the opaque
        # `PR` placeholder) so SSA versions them and downstream uses link to these
        # defs — the same SSA-tracking that P2R relies on, for the write side.
        # operands become [<pred defs...>, Rs, mask]; defs/uses split via
        # use_op_start_idx = number of predicate defs.
        elif self.opcodes[0] == "R2P":
            # The disassembler omits the mask when the encoded byte is 0xff.
            if len(self.operands) == 2 and self.operands[0].name == "PR":
                self.operands = [*self.operands, Operand.from_immediate("0xff", 0xff)]
            if (len(self.operands) >= 3 and self.operands[2].is_immediate
                    and isinstance(self.operands[2].immediate_value, int)):
                mask = self.operands[2].immediate_value
                preds = [Operand.from_reg(f"P{bit}", f"P{bit}")
                         for bit in range(7) if mask & (1 << bit)]
                if preds:
                    self.operands = preds + self.operands[1:]
                    self.use_op_start_idx = len(preds)

        # MUFU.RCP64H / MUFU.RSQ64H: R6:R5 is 64-bit source pair
        elif self.opcodes[0] == "MUFU" and len(self.opcodes) > 1 and self.opcodes[1] in ("RCP64H", "RSQ64H"):
            self.operands[1] = Operand.create_reg_pair(self.operands[1])
        
        # LEPC R20 ; R21:R20 is 64-bit destination (PC value)
        elif self.opcodes[0] == "LEPC":
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
        
        # CALL.ABS.NOINC R14 ; R15:R14 is 64-bit call target
        elif self.opcodes[0] == "CALL" and "ABS" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
        
        # I2F.F64.U16 R2, R10 ; R3:R2 is 64-bit destination
        # I2F.F64.S64 R16, R4 ; R17:R16 dest, R5:R4 source
        elif self.opcodes[0] == "I2F":
            modifiers = set(self.opcodes[1:])
            if any("64" in m and m.startswith("F") for m in modifiers):
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if any(m in ("S64", "U64") for m in modifiers):
                self.operands[1] = Operand.create_reg_pair(self.operands[1])
        
        # F2I.U32.F64.TRUNC R3, R2 ; R3:R2 source is 64-bit
        # F2I.U64.TRUNC R22, R22 ; R23:R22 dest is 64-bit
        elif self.opcodes[0] == "F2I":
            modifiers = set(self.opcodes[1:])
            if any(m in ("U64", "S64") for m in modifiers):
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if "F64" in modifiers:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])
        
        # F2F.F64.F32 R6, R7 ; dest_type and src_type are the first two
        # recognised float tokens after the base opcode.  Skip modifiers
        # like SATFINITE.
        elif self.opcodes[0] in ("F2F", "F2FP") and len(self.opcodes) > 2:
            _FT = {"F16", "F32", "F64", "BF16", "E5M2", "E4M3"}
            _ftypes = [t for t in self.opcodes[1:] if t in _FT]
            dest_ft = _ftypes[0] if len(_ftypes) > 0 else ""
            src_ft = _ftypes[1] if len(_ftypes) > 1 else ""
            if "64" in dest_ft:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if "64" in src_ft:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])
        
        # ATOMG.E.ADD.STRONG.GPU PT, R2, desc[UR6][R2.64], R9 ;
        # ATOMG.E.XOR.64.STRONG.GPU PT, RZ, desc[UR4][R10.64], R2 ;
        # operands[2] = mem address (always 64-bit pair). For ".64" atomics the
        # old-value def (operands[1]) and the source value operand(s) (>=3) are
        # ALSO 64-bit reg pairs — without this the high word of the atomic
        # value is dropped (only low 32 bits XOR/add). CAS has cmp+new sources.
        elif self.opcodes[0] == "ATOMG" and "E" in self.opcodes:
            if len(self.operands) > 2 and self.operands[2].is_reg:
                self.operands[2] = Operand.create_reg_pair(self.operands[2])
            if "64" in self.opcodes:
                if len(self.operands) > 1 and self.operands[1].is_reg:
                    self.operands[1] = Operand.create_reg_pair(self.operands[1])
                for vi in range(3, len(self.operands)):
                    if self.operands[vi].is_reg:
                        self.operands[vi] = Operand.create_reg_pair(self.operands[vi])

        # ATOM.E.ADD.F16x2.RN.STRONG.GPU P0, RZ, [R2], R5 ;
        # The memory address operand (operands[2]) needs 64-bit register pair
        # expansion; ".64" variants also pair the def + source value (see ATOMG).
        elif self.opcodes[0] == "ATOM" and "E" in self.opcodes:
            if len(self.operands) > 2 and self.operands[2].is_reg:
                self.operands[2] = Operand.create_reg_pair(self.operands[2])
            if "64" in self.opcodes:
                if len(self.operands) > 1 and self.operands[1].is_reg:
                    self.operands[1] = Operand.create_reg_pair(self.operands[1])
                for vi in range(3, len(self.operands)):
                    if self.operands[vi].is_reg:
                        self.operands[vi] = Operand.create_reg_pair(self.operands[vi])

        # REDG.E.ADD desc[UR6][R2.64], R5 ;
        # operands[0] = mem address (64-bit pair). REDG has no def; the source
        # value is operands[1] and is a 64-bit pair for ".64" reductions.
        elif self.opcodes[0] == "REDG" and "E" in self.opcodes:
            if len(self.operands) > 0 and self.operands[0].is_reg:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            # REDG.E.ADD.F32x{2,4} (FA3 bwd dQ_accum) reduces a VECTOR of 2/4 f32
            # from consecutive source registers (R, R+1, ...) to consecutive gmem
            # words. Expand the source operand into the matching reg group so def-use
            # links all N accumulator regs (else R+1.. read 0 -> only 1 of N stored).
            _redg_mods = [t.upper() for t in self.opcodes[1:]]
            _vw = 4 if "F32X4" in _redg_mods else (2 if "F32X2" in _redg_mods else 1)
            if _vw > 1 and len(self.operands) > 0 and self.operands[-1].is_reg \
                    and not self.operands[-1].is_reg_group:
                self.operands[-1] = Operand.create_reg_group(self.operands[-1], _vw)
            # Pair the 64-bit VALUE operand. The width token may be a bare "64"
            # OR a typed "F64"/"S64"/"U64" (e.g. REDG.E.ADD.F64.RN.STRONG.GPU) —
            # a plain `"64" in opcodes` membership test misses the typed forms,
            # so the f64 accumulator's low word alone was taken and fpext'd as
            # f32 -> garbage/NaN (gpp, ccsd-trpdrv).
            if any(t in ("64", "F64", "S64", "U64") for t in self.opcodes):
                for vi in range(1, len(self.operands)):
                    if self.operands[vi].is_reg:
                        self.operands[vi] = Operand.create_reg_pair(self.operands[vi])

        # LDGSTS.E [R7], desc[UR6][R2.64] ;
        # operand[0] = shared addr [R7] (32-bit), operand[1] = global addr desc[UR6][R2.64] (64-bit pair)
        elif self.opcodes[0] == "LDGSTS" and "E" in self.opcodes:
            if len(self.operands) > 1 and self.operands[1].is_reg:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # SYNCS.EXCH.64 takes the barrier state from a register pair. Track
        # its high word as a use so SSA preserves the initial phase bit.
        elif self.opcodes == ["SYNCS", "EXCH", "64"] and len(self.operands) == 3:
            self.operands[2] = Operand.create_reg_pair(self.operands[2])

        # STAS [R6.64+0x10], R8 uses R6+0x10 as a DSMEM destination and
        # R7 as the completion barrier. The .64 operand is two addresses.
        elif self.opcodes == ["STAS"] and len(self.operands) == 2:
            dst = self.operands[0]
            if dst.is_mem_addr and dst.suffix == "64" and re.fullmatch(r"R\d+", dst.reg or ""):
                barrier = f"R{int(dst.reg[1:]) + 1}"
                dst.suffix = None
                self.operands.append(Operand.from_reg(barrier, barrier))

        # UBLKCP.S.G [UR6], [UR4], UR10 ;  (cp.async.bulk global→shared)
        # operand[0] = base of a {dst, mbar} TUPLE — like UTMALDG, the smem
        #   dst offset is URn and the completion MBARRIER address is URn+1
        #   (bwd SASS: `UMOV UR7, UR17` stages the barrier right before
        #   `UBLKCP [UR6], ...`). Synthesize the member so the nvvm lowering
        #   can emit cp.async.bulk's mbarrier operand and its producer chain
        #   survives DCE. (32-bit window addrs, cluster-rank bits 24+.)
        # operand[1] = global src address (implicit 64-bit UR pair, no .64
        # suffix in SASS), operand[2] = copy size in 16-byte units.
        elif self.opcodes[0] == "UBLKCP":
            if len(self.operands) > 1 and self.operands[1].is_reg:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])
            base_reg = getattr(self.operands[0], 'reg', None)
            if base_reg and base_reg.startswith('UR'):
                try:
                    b = int(base_reg[2:])
                    m_reg = f"UR{b + 1}"
                    m_op = Operand.from_reg(m_reg, m_reg)
                    m_op.tma_tuple_idx = 1  # mbarrier member
                    self.operands.append(m_op)
                except ValueError:
                    pass

        # UBLKRED.G.S.ADD.F32.RN [UR28], [UR18], UR19, desc[UR26] ;
        #   (TMA bulk-REDUCE-add global += shared, fp32; FA3 bwd dQ accumulation)
        # operand[0] = global DST address (implicit 64-bit UR pair, no .64 suffix),
        # operand[1] = shared SRC byte offset (32-bit single UR),
        # operand[2] = reduce size in 16-byte units (single UR),
        # operand[3] = desc[URn] access descriptor (64-bit UR pair, like UTMASTG).
        elif self.opcodes[0] == "UBLKRED":
            if len(self.operands) >= 1 and self.operands[0].is_reg:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if len(self.operands) >= 4:
                self.operands[3] = self._widen_tma_desc_pair(self.operands[3])

        # ST.E.SYS [R2], R5 ;
        elif "ST" in self.opcodes[0] and "E" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if len(self.operands) > 1 and "128" in self.opcodes:
                self.operands[1] = Operand.create_reg_quad(self.operands[1])
            elif len(self.operands) > 1 and "64" in self.opcodes:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])
            
        # LD.E.SYS R0, [R2] ; 
        elif "LD" in self.opcodes[0] and "E" in self.opcodes:
            self.operands[1] = Operand.create_reg_pair(self.operands[1])
            if len(self.operands) > 1 and "128" in self.opcodes:
                self.operands[0] = Operand.create_reg_quad(self.operands[0])
            elif len(self.operands) > 1 and "64" in self.opcodes:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])
            
        # STG.E.64.SYS [R2+0x70f800], R4 ; 
        elif "STG" in self.opcodes[0] and "E" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])
            if len(self.operands) > 1 and "128" in self.opcodes:
                self.operands[1] = Operand.create_reg_quad(self.operands[1])
            elif len(self.operands) > 1 and "64" in self.opcodes:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # LDG.E.64.SYS R4 = [R2] defines R4 and R5
        elif "LDG" in self.opcodes[0] and "E" in self.opcodes:
            self.operands[1] = Operand.create_reg_pair(self.operands[1])
            if len(self.operands) > 1 and "128" in self.opcodes:
                self.operands[0] = Operand.create_reg_quad(self.operands[0])
            elif len(self.operands) > 1 and "64" in self.opcodes:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # LDC.64 / ULDC.64: loads 64-bit into register pair (R2:R3 or UR6:UR7)
        elif self.opcodes[0] in ("LDC", "ULDC", "LDCU") and "128" in self.opcodes:
            self.operands[0] = Operand.create_reg_quad(self.operands[0])
        elif self.opcodes[0] in ("LDC", "ULDC", "LDCU") and "64" in self.opcodes:
            dest = self.operands[0]
            if dest.is_reg and not dest.is_const_mem:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # LDL.64 R12 = [R52+0x110] defines R12 and R13
        elif self.opcodes[0] == "LDL" and "64" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # STL.64 [R52+0x110] = R12 uses R12 and R13
        elif self.opcodes[0] == "STL" and "64" in self.opcodes:
            self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # LDL.128 R12 = [R52+0x110] loads 128-bit into R12, R13, R14, R15
        elif self.opcodes[0] == "LDL" and "128" in self.opcodes:
            self.operands[0] = Operand.create_reg_quad(self.operands[0])

        # STL.128 [R10+0x100], R87 stores 128-bit from R87, R88, R89, R90
        elif self.opcodes[0] == "STL" and "128" in self.opcodes:
            self.operands[1] = Operand.create_reg_quad(self.operands[1])

        elif (self.opcodes == ["UTCIMMA"] and len(self.operands) == 6
              and all(re.fullmatch(r"gdesc\[UR\d+\]", op.name or "") for op in self.operands[:2])
              and all(re.fullmatch(r"tmem\[UR\d+\]", op.name or "") for op in self.operands[2:4])
              and (idesc := re.fullmatch(r"idesc\[(UR\d+)\]", self.operands[4].name or ""))
              and (self.operands[5].is_predicate_reg or self.operands[5].is_pt)):
            self.operands[:2] = [Operand.create_reg_pair(op) for op in self.operands[:2]]
            self.operands[4] = Operand.from_reg(idesc[1], idesc[1])

        elif self.opcodes[0] == "LDTM" and self.ldtm_register_count:
            count = self.ldtm_register_count
            self.use_op_start_idx = count
            if len(self.operands) == 2 and count > 1 and self.operands[0].is_reg:
                self.operands[:1] = Operand.create_reg_group(self.operands[0], count).reg_group

        # LDSM.16.M88.4 R132, [R99] loads 4 regs from shared memory
        elif self.opcodes[0] == "LDSM":
            count = int(self.opcodes[-1]) if self.opcodes[-1].isdigit() else 1
            if count == 4:
                self.operands[0] = Operand.create_reg_quad(self.operands[0])
            elif count == 2:
                self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # STSM.16.M88.4 [R99], R132 stores 4 regs to shared memory
        elif self.opcodes[0] == "STSM":
            count = int(self.opcodes[-1]) if self.opcodes[-1].isdigit() else 1
            if count == 4:
                self.operands[1] = Operand.create_reg_quad(self.operands[1])
            elif count == 2:
                self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # LDS.U.64 R181 = [R66] loads 64-bit into R181:R182
        elif self.opcodes[0] == "LDS" and "64" in self.opcodes:
            self.operands[0] = Operand.create_reg_pair(self.operands[0])

        # LDS.U.128 R188 = [R61+0x100] loads 128-bit into R188, R189, R190, R191
        elif self.opcodes[0] == "LDS" and "128" in self.opcodes:
            self.operands[0] = Operand.create_reg_quad(self.operands[0])

        # STS.64 [R66], R168 stores 64-bit from R168:R169
        elif self.opcodes[0] == "STS" and "64" in self.opcodes:
            self.operands[1] = Operand.create_reg_pair(self.operands[1])

        # STS.128 [R66], R168 stores 128-bit from R168, R169, R170, R171
        elif self.opcodes[0] == "STS" and "128" in self.opcodes:
            self.operands[1] = Operand.create_reg_quad(self.operands[1])

        # SM90 TMA tensor copy, all observed SASS spellings (FA3 cubins):
        #   UTMALDG.ND           [URt], [URd]                  (tma_min)
        #   UTMALDG.ND           [URt], [URd], desc[URm]       (FA fwd Q/K/V)
        #   UTMALDG.ND.MULTICAST [URt], [URd], URmask, desc[URm]
        #   UTMASTG.ND           [URt], [URd]                  (FA epilogue O)
        # Roles (evidence: FA fwd hdim128 — URd chains from ULDC.64
        # c[0x0][0x198] + tensormap offset in Params; desc[URm] is the UMOV'd
        # {0x0, 0x14f00000} default access descriptor, NOT the tensormap):
        #   [URt]     = base of a UR TUPLE {smem, mbarrier, coords...} (load)
        #               or {smem, coords...} (store; no mbarrier). Coords are
        #               ALWAYS inline in the tuple — no coords-in-memory form.
        #   [URd]     = the 64-bit generic address of the CUtensorMap. Widen
        #               to its URd:URd+1 pair HERE (parse, before SSA/RegRemap)
        #               so both halves version together and _get_val packs one
        #               i64; else the pointer truncates to its low word.
        #   URmask    = (MULTICAST) 16-bit CTA mask; plain reg use, untouched.
        #   desc[URm] = cache/access memdesc — widened for dataflow but tagged
        #               tma_memdesc so the lifter can EXCLUDE it from role
        #               selection (mistaking it for the tensormap made every
        #               TMA read tensormap @ 0x14f0000000000000 -> fault).
        # Tuple members are synthesized as extra uses (the desc_hi pattern) so
        # SSA/DCE keep their producer chains alive and the nvvm lifter can read
        # each field by tag. (Verified on tma_min: the paired
        # SYNCS.ARRIVE.TRANS64 completes on [URt+1] = the mbarrier member.)
        elif self.opcodes[0] in ("UTMALDG", "UTMASTG") and len(self.operands) >= 2:
            for i, op in enumerate(self.operands):
                if i == 0:
                    continue  # the tuple base stays a single 32-bit member
                if getattr(op, 'desc_reg', None) is not None and not op.is_reg:
                    widened = self._widen_tma_desc_pair(op)
                    widened.tma_memdesc = True
                    self.operands[i] = widened
                elif getattr(op, 'is_mem_addr', False):
                    self.operands[i] = Operand.create_reg_pair(op)
            nd = 2
            for tok in self.opcodes[1:]:
                if tok.endswith("D") and tok[:-1].isdigit():
                    nd = int(tok[:-1])
                    break
            base_reg = getattr(self.operands[0], 'reg', None)
            is_store = self.opcodes[0] == "UTMASTG"
            n_members = nd if is_store else (1 + nd)
            if base_reg and base_reg.startswith('UR'):
                try:
                    b = int(base_reg[2:])
                    for j in range(1, n_members + 1):
                        m_reg = f"UR{b + j}"
                        m_op = Operand.from_reg(m_reg, m_reg)
                        # load: 1 = mbarrier, 2.. = coords; store: 1.. = coords
                        m_op.tma_tuple_idx = j + (1 if is_store else 0)
                        self.operands.append(m_op)
                except ValueError:
                    pass

        elif len(self.opcodes) > 1 and self.opcodes[0] == "HMMA":
            # HMMA.<shape>.<accum> D, A, B, C
            shape = self.opcodes[1] if len(self.opcodes) > 1 else ""
            accum = self.opcodes[2] if len(self.opcodes) > 2 else ""

            if shape == "884":
                d_count, a_count, b_count, c_count = 2, 2, 2, 2
            elif shape == "1688" and accum == "F16":
                d_count, a_count, b_count, c_count = 2, 2, 1, 2
            elif shape == "1688" and accum == "F32" and "TF32" in self.opcodes:
                # HMMA.1688.F32.TF32: one tf32 per full 32-bit reg (cute
                # MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>: ARegisters=
                # uint32_t[4], BRegisters=uint32_t[2]) — A 16x8 needs 4 regs,
                # B 8x8 needs 2, unlike the packed-16-bit F16/BF16 forms.
                d_count, a_count, b_count, c_count = 4, 4, 2, 4
            elif shape == "1688" and accum == "F32":
                d_count, a_count, b_count, c_count = 4, 2, 1, 4
            elif shape == "16816" and accum == "F16":
                # HMMA.16816.F16: m16n8k16 FP16 accum — D/C = 2 regs (v2f16)
                d_count, a_count, b_count, c_count = 2, 4, 2, 2
            elif shape == "16816":
                # HMMA.16816.F32: m16n8k16 — A needs 4 regs (8 FP16), B needs 2 regs (4 FP16)
                d_count, a_count, b_count, c_count = 4, 4, 2, 4
            else:
                d_count, a_count, b_count, c_count = 4, 2, 2, 4

            DRegs = Operand.create_reg_group(self.operands[0], d_count).reg_group
            ARegs = Operand.create_reg_group(self.operands[1], a_count).reg_group
            BRegs = Operand.create_reg_group(self.operands[2], b_count).reg_group
            CRegs = Operand.create_reg_group(self.operands[3], c_count).reg_group

            self.operands = DRegs + ARegs + BRegs + CRegs

        elif self.opcodes == ["FFMA2"] and len(self.operands) == 4 and self.use_op_start_idx == 2:
            expanded = Operand.create_reg_group(self.operands[0], 2).reg_group
            for op in self.operands[1:]:
                form = op.suffix.upper()
                if form == "F32":
                    lanes = [op.clone(), op.clone()]
                else:
                    lanes = Operand.create_reg_group(op, 2).reg_group
                    if "LO_HI" in form:
                        lanes.reverse()
                for lane, word in enumerate(lanes):
                    # NP negates the second destination lane after selecting its source.
                    negative = bool(op.is_negative_reg) ^ (form.endswith(".NP") and lane == 1)
                    word.prefix = "-" if negative else None
                expanded += lanes
            self.operands = expanded

        elif self.opcodes[:3] == ["QMMA", "16832", "F32"] and len(self.operands) in (4, 5):
            # Per-lane D/A/B/C register fragments of dense FP8 m16n8k32.
            expanded = []
            for operand, count in zip(self.operands, (4, 4, 2, 4)):
                expanded.extend(Operand.create_reg_group(operand, count).reg_group)
            self.operands = expanded + self.operands[4:]

        # IMMA.8816.S8.S8 D, A, B, C — integer MMA
        # m8n8k16: D -> 2 regs, A -> 1 reg, B -> 1 reg, C -> 2 regs
        # m16n8k32: D -> 4 regs, A -> 4 regs, B -> 2 regs, C -> 4 regs
        # IMMA.SP.16864.S8.S8 D, A, B, C, meta, selector_imm — sparse 2:4
        # m16n8k64 sparse: D -> 4, A -> 4 (sparse-compressed), B -> 4, C -> 4
        # + 1 metadata register + 1 selector immediate.
        elif len(self.opcodes) > 1 and self.opcodes[0] == "IMMA":
            is_sparse = self.opcodes[1] == "SP"
            shape = self.opcodes[2] if is_sparse else self.opcodes[1]
            if is_sparse and shape == "16864":
                d_count, a_count, b_count, c_count = 4, 4, 4, 4
            elif shape in ("8816", "8832"):
                d_count, a_count, b_count, c_count = 2, 1, 1, 2
            elif shape == "16832":
                d_count, a_count, b_count, c_count = 4, 4, 2, 4
            elif shape == "16816":
                d_count, a_count, b_count, c_count = 4, 2, 1, 4
            else:
                d_count, a_count, b_count, c_count = 4, 1, 1, 4
            DRegs = Operand.create_reg_group(self.operands[0], d_count).reg_group
            ARegs = Operand.create_reg_group(self.operands[1], a_count).reg_group
            BRegs = Operand.create_reg_group(self.operands[2], b_count).reg_group
            CRegs = Operand.create_reg_group(self.operands[3], c_count).reg_group
            new_ops = DRegs + ARegs + BRegs + CRegs
            # Sparse IMMA appends a metadata register and a selector immediate
            # (operands[4:]). Keep them verbatim so the lifter can reach them
            # via get_uses() after the A/B/C block.
            if is_sparse:
                new_ops += list(self.operands[4:])
            self.operands = new_ops

        # DMMA.<shape> D, A, B, C — F64 tensor-core MMA
        # Per-thread f64 fragment counts (PTX mma.sync .f64 / cute MMA_Traits):
        #   m8n8k4   → D2 A1 B1 C2
        #   m16n8k8  → D4 A4 B2 C4
        #   m16n8k16 → D4 A8 B4 C4
        # Each f64 is a register pair (2×r32). Expand to N separate reg-pair
        # Operands rather than one flat group, matching DFMA/DADD semantics so
        # TypeAnalysis can tag them FLOAT64 via the reg_pair lattice widening.
        elif len(self.opcodes) > 1 and self.opcodes[0] == "DMMA":
            _dmma_shape = self.opcodes[1]
            if _dmma_shape == "16x8x16":
                d_pairs, a_pairs, b_pairs, c_pairs = 4, 8, 4, 4
            elif _dmma_shape == "16x8x8":
                d_pairs, a_pairs, b_pairs, c_pairs = 4, 4, 2, 4
            else:  # 8x8x4
                d_pairs, a_pairs, b_pairs, c_pairs = 2, 1, 1, 2

            def _dmma_pairs(op, n_pairs):
                if not op.is_reg or "RZ" in op.reg:
                    return [Operand.create_reg_pair(op) for _ in range(n_pairs)]
                base = int(op.reg[1:]) if op.reg[0] == 'R' else None
                if base is None:
                    return [Operand.create_reg_pair(op) for _ in range(n_pairs)]
                pairs = []
                for i in range(n_pairs):
                    stub = Operand.from_reg(f"R{base + 2 * i}",
                                            f"R{base + 2 * i}")
                    pairs.append(Operand.create_reg_pair(stub))
                return pairs

            Dpairs = _dmma_pairs(self.operands[0], d_pairs)
            Apairs = _dmma_pairs(self.operands[1], a_pairs)
            Bpairs = _dmma_pairs(self.operands[2], b_pairs)
            Cpairs = _dmma_pairs(self.operands[3], c_pairs)
            self.operands = Dpairs + Apairs + Bpairs + Cpairs

        # HGMMA/WGMMA/QGMMA: warpgroup MMA (SM90)
        # Register form:    HGMMA.<shape>.<accum> D, A, B, C  (4 operands)
        # Descriptor form:  HGMMA.<shape>.<accum> D, gdesc[URx].tnspA.tnspB, C [, !UPT, gsb0]
        # QGMMA (FP8) shares the identical operand layout — only opcode tokens differ.
        elif len(self.opcodes) > 1 and self.opcodes[0] in ("HGMMA", "WGMMA", "QGMMA"):
            shape_str = self.opcodes[1]  # e.g. "64x8x16"
            accum = self.opcodes[2] if len(self.opcodes) > 2 else "F32"
            d_count = self._wgmma_d_reg_count(shape_str, accum)
            c_count = d_count  # C has same register count as D

            # Detect descriptor vs register form by checking operand content.
            # Two descriptor layouts exist (ptxas picks based on A's source):
            #   SS form: D, gdesc[A], C [, !UPT, gsb0]        — A and B both from
            #            shared-mem descriptors; gdesc at operand index 1.
            #   RS form: D, A_reg, gdesc[B], C [, !UPT, gsb0] — A from registers,
            #            B from a shared-mem descriptor; gdesc at operand index 2.
            # Real CUTLASS / FlashAttention mainloops emit BOTH. Detect by
            # scanning all operands for the gdesc marker rather than assuming the
            # descriptor sits at index 1 (which dropped the RS form into the
            # register-form branch and silently grabbed the descriptor address
            # arithmetic as a bogus "B fragment").
            desc_idx = None
            for i, op in enumerate(self.operands):
                if hasattr(op, 'name') and 'gdesc' in str(op.name):
                    desc_idx = i
                    break
            is_descriptor = desc_idx is not None

            if is_descriptor:
                desc_op = self.operands[desc_idx]
                DRegs = Operand.create_reg_group(self.operands[0], d_count).reg_group

                # GMMA descriptor PAIR recovery (SS form).
                # The SS-form operand `gdesc[URn]` names only URn (the A
                # descriptor low word, whose start_address field = A's shared
                # byte offset >> 4). The wgmma.mma_async takes TWO 64-bit matrix
                # descriptors (desc_a, desc_b); ptxas lays them out as register
                # pairs, so the B descriptor's low word lives in UR(n+2) — it is
                # implied by convention, NOT spelled on the operand. (UR(n+1) is
                # the high word of desc_a — stride/layout bits — and ptxas often
                # reuses it for unrelated values after the build, so it is NOT
                # the B base.) Without naming UR(n+2), the B address-build chain
                # (UPRMT/USHF/ULOP3 producing B's base) has no use and is deleted
                # by DCE, so the lifter can never recover B and the product
                # collapses to the C passthrough.
                #
                # Synthesize the HI (B) descriptor operand UR(n+2) and carry it
                # as a use right after the descriptor. This keeps B's chain live
                # and gives the lifter a fetchable name for the B-base. Tagged
                # desc_hi=True so the lifter can pick it out.
                desc_hi_op = None
                desc_a_hi_op = None
                desc_b_hi_op = None
                # The B descriptor lives at {URn+2, URn+3} in ALL forms —
                # SS, .tnspB RS, and plain RS. (Runtime-proven bit-exact for
                # SS + .tnspB RS on FA3 fwd; for plain RS the FA3 bwd hdim96
                # mainloop shows the same shape: the k-slice desc-lo steps
                # land in UR14=UR12+{0,2,0x100,...} (SW64 K-traversal) with
                # `UMOV UR15, 0x80000020` as the layout hi — {n+2, n+3}.
                # Experiments with {n,n+1} / {n+2,n+1} both produced NaNs.)
                lo_reg = getattr(desc_op, 'desc_reg', None) or getattr(desc_op, 'reg', None)
                if lo_reg and lo_reg.startswith('UR'):
                    try:
                        lo_num = int(lo_reg[2:])
                        hi_reg = f"UR{lo_num + 2}"
                        desc_hi_op = Operand.from_reg(hi_reg, hi_reg)
                        desc_hi_op.desc_reg = hi_reg
                        desc_hi_op.desc_hi = True
                        # UR(n+1) is desc_a's HIGH word, carrying the layout_type
                        # (swizzle mode, bits[63:62] = high-word bits[31:30]) and
                        # the leading/stride byte offsets. Carry it so the lifter
                        # can distinguish SW128 (B128) from no-swizzle (K-INTER)
                        # and recover the tile K extent. Keeping it alive also
                        # prevents DCE from dropping the high-word build chain.
                        ahi_reg = f"UR{lo_num + 1}"
                        desc_a_hi_op = Operand.from_reg(ahi_reg, ahi_reg)
                        desc_a_hi_op.desc_reg = ahi_reg
                        desc_a_hi_op.desc_a_hi = True
                        # desc_B's HIGH word (UR(n+3)) carries the B operand's
                        # swizzle mode. The x86warp path must NOT keep it live
                        # (its build chain would survive DCE and shift SSA
                        # value-numbering for every SW128 kernel, breaking the
                        # verified byte-identity); it reads the layout immediates
                        # statically via the RecordWgmmaSwizzle transform. The
                        # NVVM path re-emits real wgmma.mma_async and needs BOTH
                        # full 64-bit descriptors as live values — gate the
                        # synthesis on the backend (main.py sets SYNTH_DESC_B_HI
                        # for --lifter nvvm before parsing).
                        if Instruction.SYNTH_DESC_B_HI:
                            bhi_reg = f"UR{lo_num + 3}"
                            desc_b_hi_op = Operand.from_reg(bhi_reg, bhi_reg)
                            desc_b_hi_op.desc_reg = bhi_reg
                            desc_b_hi_op.desc_b_hi = True
                    except ValueError:
                        desc_hi_op = None
                        desc_a_hi_op = None
                        desc_b_hi_op = None

                # Register-source-A operands precede the descriptor (operands
                # [1 .. desc_idx-1]); for m64 A is a 4-reg fragment. Tag them
                # so the nvvm lifter can (a) detect the RS form and (b) fetch
                # the fragment values — untagged they are indistinguishable
                # from C accumulator regs in the use list. NOTE: the RS form
                # uses the SAME descriptor-window convention as SS — the B
                # descriptor is {URn+2, URn+3} (desc_hi/desc_b_hi tags below);
                # {URn, URn+1} is the A-desc slot, unused in RS and typically
                # holding a STALE previous gemm's descriptor (runtime-proven
                # on FA3: reading it fed the P@V gemm the QK gemm's Q-desc).
                a_ops = []
                if desc_idx >= 2:
                    a_ops = Operand.create_reg_group(self.operands[1], 4).reg_group
                    for a_op in a_ops:
                        a_op.wgmma_a_frag = True

                # Trailing operands after the descriptor: the C accumulator
                # (first register) followed by non-register flags (!UPT, gsb0).
                remaining = self.operands[desc_idx + 1:]
                c_op = None
                extra_ops = []
                for op in remaining:
                    if c_op is None and op.is_reg:
                        c_op = op
                    else:
                        extra_ops.append(op)

                if c_op is not None and c_op.reg != "RZ":
                    CRegs = Operand.create_reg_group(c_op, c_count).reg_group
                else:
                    # RZ or missing C — use RZ for all C regs
                    c_op_base = c_op if c_op is not None else Operand.from_reg("RZ", "RZ")
                    CRegs = [c_op_base.clone() for _ in range(c_count)]
                # Canonical order: D regs, A regs (RS only), descriptor [, hi
                # descriptor], C regs, flags.
                desc_ops = [desc_op] + ([desc_hi_op] if desc_hi_op is not None else []) \
                           + ([desc_a_hi_op] if desc_a_hi_op is not None else []) \
                           + ([desc_b_hi_op] if desc_b_hi_op is not None else [])
                self.operands = DRegs + a_ops + desc_ops + CRegs + extra_ops
            else:
                # Register form: D, A, B, C
                a_count = 4   # A fragment: 4 regs for m64 (per-warp m16)
                parts = shape_str.split("x")
                n_dim = int(parts[1]) if len(parts) > 1 else 8
                b_count = max(n_dim // 4, 2)  # B regs: N/4, minimum 2

                if len(self.operands) >= 4:
                    DRegs = Operand.create_reg_group(self.operands[0], d_count).reg_group
                    ARegs = Operand.create_reg_group(self.operands[1], a_count).reg_group
                    BRegs = Operand.create_reg_group(self.operands[2], b_count).reg_group
                    CRegs = Operand.create_reg_group(self.operands[3], c_count).reg_group
                    self.operands = DRegs + ARegs + BRegs + CRegs

        # temp solution: add parent field to operand here
        for op in self.operands:
            # if hasattr(op, 'Parent'):
            #     raise Exception("Operand parent already set")
            op.Parent = self

    @staticmethod
    def _widen_tma_desc_pair(op):
        """Widen a UTMALDG/UTMASTG descriptor operand to its URn:URn+1 pair.

        TMA descriptor addresses are 64-bit and come in two SASS spellings:
          * plain memory form `[URn]` — the reg sits in `op.reg`, so the generic
            create_reg_pair handles it (URn:URn+1).
          * descriptor form `desc[URn]` — the parser stores URn in `op.desc_reg`
            with `op.reg=None` and `op.is_reg=False`, so create_reg_pair returns
            it UNCHANGED (its `is_reg` guard bails). That dropped the HIGH word
            (URn+1), collapsing the 64-bit descriptor pointer to its low 32 bits.
            In real FlashAttention the descriptor is `UMOV URn,0; UMOV URn+1,<hi>;
            ... desc[URn]` — losing URn+1 made `_get_val` resolve desc_addr to the
            literal low word (0) so no tensor data moved. Build the explicit pair
            from desc_reg here (parse time, before SSA/RegRemap) so both halves are
            versioned together and _get_val packs the full i64.
        """
        desc_reg = getattr(op, 'desc_reg', None)
        if desc_reg is not None and not op.is_reg and "RZ" not in desc_reg:
            is_ur = "UR" in desc_reg
            prefix_len = 2 if is_ur else 1
            prefix_str = "UR" if is_ur else "R"
            base = int(desc_reg[prefix_len:])
            lo, hi = f"{prefix_str}{base}", f"{prefix_str}{base + 1}"
            pair = Operand.from_reg_pair(f"{hi}:{lo}", lo, hi)
            pair.is_mem_addr = op.is_mem_addr
            pair.desc_reg = desc_reg
            return pair
        return Operand.create_reg_pair(op)

    def clone(self):
        cloned_operands = [op.clone() for op in self.operands]
        cloned_pflag = self.pflag.clone() if self.pflag else None
        cloned_inst = Instruction(
            id=self.id,
            opcodes=self.opcodes.copy(),
            operands=cloned_operands,
            parentBB=self.parent,
            pflag=cloned_pflag
        )
        # __init__ re-runs the operand-expansion logic (reg-group widening
        # for MMA/load/store/IMAD.WIDE/CS2R/... plus the use_op_start_idx
        # computation). `self.operands` is ALREADY expanded, so re-running
        # the expansion double-mutates it — e.g. an IMMA atom's flat
        # [D0..3, A0..3, B0..3, C0..3, meta, sel] has each of its first four
        # operands re-widened into 4-reg groups, producing sliding-overlap
        # windows (14 uses -> 26) when HandleCall inlines the atom. Restore
        # the source's already-correct operands + def/use boundary so
        # clone() is a faithful copy regardless of __init__ re-mutation.
        cloned_inst.operands = cloned_operands
        cloned_inst.use_op_start_idx = self.use_op_start_idx
        cloned_inst.inst_content = self.inst_content
        cloned_inst.ctl_code = self.ctl_code
        return cloned_inst

    @staticmethod
    def _wgmma_d_reg_count(shape_str, accum):
        """Compute the number of D/C registers for WGMMA/HGMMA.

        WGMMA operates on a warp group (4 warps, M=64).  Per-thread register
        counts for D/C depend on N and accumulator type:
          F32: d_count = N / 2  (each reg holds 1 FP32, matrix is M_per_warp x N = 16 x N)
          F16: d_count = N / 4  (each reg holds 2 FP16)

        Shape string format: "64xNxK" (e.g. "64x8x16", "64x128x16").
        """
        parts = shape_str.split("x")
        n_dim = int(parts[1]) if len(parts) > 1 else 8
        if accum.startswith("F16"):
            return max(n_dim // 4, 2)
        else:
            # F32, F32.BF16, etc.
            return max(n_dim // 2, 4)

    def set_ctl_code(self, CtlCode):
        if self.ctl_code != None:
            raise PresetCtlCodeException

        self.ctl_code = CtlCode
        
    def is_exit(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "EXIT"

    def is_branch(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] == "BRA" or self.opcodes[0] == "BRX" 
                                          or self.opcodes[0] == "PBRA" or self.opcodes[0] == "SWITCH")
    
    def is_indirect_branch(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "BRX"
    
    def is_return(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "RET"

    def is_call(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "CALL"
    
    def is_set_predicate(self):
        return self.opcodes[0] == "ISETP"
    
    def is_phi(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] == "PHI" or self.opcodes[0] == "PHI64")
        
    def predicated(self):
        return self.pflag is not None

    def is_conditional_branch(self):
        return self.is_branch() and self.predicated()

    def in_cond_path(self):
        return self.predicated()
    
    def is_nop(self):
        return len(self.opcodes) > 0 and self.opcodes[0] == "NOP"

    def is_addr_compute(self):
        if len(self.opcodes) > 0 and self.opcodes[0] == "IADD":
            # Check operands
            if len(self.operands) == 3:
                operand = self.operands[2]
            # Check function argument operand
                return operand.is_arg

        return False

    @property
    def ldtm_register_count(self):
        modifiers = self.opcodes[1:]
        if not modifiers:
            return 1
        if len(modifiers) == 1 and modifiers[0] in ("x2", "x4", "x8", "x16", "x32", "x64", "x128"):
            return int(modifiers[0][1:])
        return 0

    def is_load(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["LDG", "LD", "LDS", "LDC", "ULDC", "LDL", "LDSM", "LDTM"])
    
    def is_global_load(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["LDG"])

    def is_store(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["STG", "SUST", "ST", "STS", "STL", "LDGSTS", "STSM", "STAS"])
    
    def is_global_store(self):
        return len(self.opcodes) > 0 and (self.opcodes[0] in ["STG"])
           
    def get_reg_name(self, Reg):
        return Reg.split('@')[0]
    
    def rename_reg(self, Reg, Inst):
        reg_name = self.get_reg_name(Reg)
        new_reg = reg_name + "@" + str(Inst.id)
        return new_reg
    
    # Get def operand
    def get_defs(self):
        return self.operands[:self.use_op_start_idx]
    
    def get_def_by_reg(self, Reg):
        for defOp in self.get_defs():
            if defOp.reg == Reg:
                return defOp
        return None

    # Get use operand
    def get_uses(self):
        return self.operands[self.use_op_start_idx:]
    
    def get_uses_with_predicate(self):
        uses = self.get_uses()
        if self.predicated():
            uses = [self.pflag] + uses
        return uses
    
    # Get branch flag
    def get_branch_flag(self):
        operand = self.operands[0]
        if operand.is_predicate_reg or operand.is_pt:
            return operand.name
        else:
            return None
    
    def __str__(self):
        pred_prefix = f"@{self.pflag} " if self.pflag else ""
        opcodes_str  = '.'.join(self.opcodes)
        def_strs  = [str(op) for op in self.get_defs()]
        use_strs  = [str(op) for op in self.get_uses()]
        operand_section = ""

        if def_strs:
            operand_section += ", ".join(def_strs)
            operand_section += " = "

        if use_strs:
            operand_section += ", ".join(use_strs)

        if operand_section:
            return f"{pred_prefix}{opcodes_str} {operand_section}"
        else:
            return f"{pred_prefix}{opcodes_str}"
    
    def __repr__(self):
        return '<' + self.__str__() + '>'

    def is_predicate_reg(self, opcode):
        if opcode[0] == 'P' and opcode[1].isdigit():
            return True
        if opcode[0] == '!' and opcode[1] == 'P' and opcode[2].isdigit():
            return True
        return False

    def dump(self):
        defs = []
        for op in self.get_defs():
            if op.is_reg_group:
                if op.is_mem_addr:
                    parts = []
                    core_name = op.reg_group[0].reg
                    if core_name:
                        parts.append(core_name)
                    if op.index_reg:
                        parts.append(f"+{op.index_reg}")
                    if op.offset_value is not None:
                        offset = int(op.offset_value)
                        if offset != 0 or not parts:
                            if offset < 0:
                                parts.append(f"-0x{abs(offset):x}")
                            else:
                                prefix_sign = "+" if parts else ""
                                parts.append(f"{prefix_sign}0x{offset:x}")
                    base_str = f"[{''.join(parts)}]"
                else:
                    base_str = op.reg_group[0].reg
                    if op.offset_value is not None:
                        offset = int(op.offset_value)
                        if offset < 0:
                            base_str += f"-0x{abs(offset):x}"
                        elif offset > 0:
                            base_str += f"+0x{offset:x}"
                if op.is_mem_addr:
                    inner = base_str[1:-1]
                    if op.suffix:
                        if op.suffix == "reuse":
                            inner += ".reuse"
                        else:
                            inner += f".{op.suffix}"
                    if op.reuse and op.suffix != "reuse":
                        inner += ".reuse"
                    base_str = f"[{inner}]"
                else:
                    if op.suffix:
                        if op.suffix == "reuse":
                            base_str += ".reuse"
                        else:
                            base_str += f".{op.suffix}"
                    if op.reuse and op.suffix != "reuse":
                        base_str += ".reuse"
                if op.prefix:
                    s = base_str
                    if "|" in op.prefix:
                        s = f"|{s}|"
                    other_prefix = op.prefix.replace("|", "")
                    if other_prefix:
                        s = f"{other_prefix}{s}"
                    base_str = s
                defs.append(base_str)
            else:
                defs.append(str(op))

        uses = []
        for op in self.get_uses():
            if op.implicit:
                continue
            if op.is_reg_group:
                if op.is_mem_addr:
                    parts = []
                    core_name = op.reg_group[0].reg
                    if core_name:
                        parts.append(core_name)
                    if op.index_reg:
                        parts.append(f"+{op.index_reg}")
                    if op.offset_value is not None:
                        offset = int(op.offset_value)
                        if offset != 0 or not parts:
                            if offset < 0:
                                parts.append(f"-0x{abs(offset):x}")
                            else:
                                prefix_sign = "+" if parts else ""
                                parts.append(f"{prefix_sign}0x{offset:x}")
                    base_str = f"[{''.join(parts)}]"
                else:
                    base_str = op.reg_group[0].reg
                    if op.offset_value is not None:
                        offset = int(op.offset_value)
                        if offset < 0:
                            base_str += f"-0x{abs(offset):x}"
                        elif offset > 0:
                            base_str += f"+0x{offset:x}"
                if op.is_mem_addr:
                    inner = base_str[1:-1]
                    if op.suffix:
                        if op.suffix == "reuse":
                            inner += ".reuse"
                        else:
                            inner += f".{op.suffix}"
                    if op.reuse and op.suffix != "reuse":
                        inner += ".reuse"
                    base_str = f"[{inner}]"
                else:
                    if op.suffix:
                        if op.suffix == "reuse":
                            base_str += ".reuse"
                        else:
                            base_str += f".{op.suffix}"
                    if op.reuse and op.suffix != "reuse":
                        base_str += ".reuse"
                if op.prefix:
                    s = base_str
                    if "|" in op.prefix:
                        s = f"|{s}|"
                    other_prefix = op.prefix.replace("|", "")
                    if other_prefix:
                        s = f"{other_prefix}{s}"
                    base_str = s
                uses.append(base_str)
            else:
                uses.append(str(op))
        operands = []
        if defs:
            operands.extend(defs)
        if uses:
            operands.extend(uses)

        # Canonicalize HMMA text form for assemblers (e.g., CuAssembler) that
        # expect HMMA.<shape>.<accum> D, A, B, C rather than expanded register lists.
        if self.opcodes and self.opcodes[0] == "HMMA":
            shape = self.opcodes[1] if len(self.opcodes) > 1 else ""
            accum = self.opcodes[2] if len(self.opcodes) > 2 else ""
            if shape == "16816":
                a_count = 4
                b_count = 2
            elif shape == "1688" and accum == "F32":
                a_count = 2
                b_count = 1
            else:
                a_count = 2
                b_count = 2
            c_idx = a_count + b_count
            if defs and len(uses) > c_idx:
                operands = [defs[0], uses[0], uses[a_count], uses[c_idx]]

        pred_prefix = f"@{self.pflag} " if self.pflag else ""
        opcodes_str = ".".join(self.opcodes)
        if operands:
            if opcodes_str == "BRA":
                inst_text = f"{pred_prefix}{opcodes_str} {', '.join(operands)};"
            else:
                inst_text = f"{pred_prefix}{opcodes_str} {', '.join(operands)} ;"
        else:
            inst_text = f"{pred_prefix}{opcodes_str} ;"

        prefix = f"        /*{self.id}*/"
        inst_col = 35
        if pred_prefix:
            inst_col = max(1, inst_col - len(pred_prefix))
        encoding_col = 82
        line = prefix + (" " * max(1, inst_col - len(prefix))) + inst_text
        if self.encoding:
            pad = max(1, encoding_col - 1 - len(line))
            line += (" " * pad) + f"/* {self.encoding} */"
        lines = [line]
        if self.ctl_code and self.ctl_code.content:
            lines.append((" " * (encoding_col - 1)) + f"/* {self.ctl_code.content} */")
        return lines
