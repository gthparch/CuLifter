SR_TID = "SR_TID"
SR_NTID = "SR_NTID"
SR_CTAID = "SR_CTAID"
SR_GRID_DIM = "SR_GRID_DIM"
SR_LANE = "SR_LANE"
SR_WARP = "SR_WARP"
SR_WARPSIZE = "SR_WARPSIZE"
SR_WARPSZ = "SR_WARPSZ"
SR_CLOCK = "SR_CLOCK"
SR_EQMASK = "SR_EQMASK"
SR_LEMASK = "SR_LEMASK"
SR_LTMASK = "SR_LTMASK"
SR_GEMASK = "SR_GEMASK"
SR_GTMASK = "SR_GTMASK"
SR_ACTIVEMASK = "SR_ACTIVEMASK"
SR_CGACTAID = "SR_CgaCtaId"
SR_SWINHI = "SR_SWINHI"
SR_SWINLO = "SR_SWINLO"

# Single-call startswith tuple for Operand.is_special_reg (hot path).
_SPECIAL_REG_PREFIXES = (
    SR_TID, SR_NTID, SR_CTAID, SR_GRID_DIM, SR_LANE, SR_WARP,
    SR_WARPSIZE, SR_WARPSZ, SR_CLOCK, SR_EQMASK, SR_LEMASK, SR_LTMASK,
    SR_GEMASK, SR_GTMASK, SR_ACTIVEMASK, SR_CGACTAID, SR_SWINHI,
    SR_SWINLO, "SR_CgaSize", "SR_VIRTUALSMID",
)

import re
import struct


class InvalidOperandException(Exception):
    pass


class Operand:
    @staticmethod
    def _extract_reuse(suffix):
        """Pull the .reuse scheduling hint out of a dot-joined suffix chain.

        nvdisasm appends .reuse to the LAST modifier of an operand, so a
        register can print as `R122.reuse.H1` (reuse + semantic .H1
        half-select). Treating the chain as one token kept "reuse.H1" as
        the suffix, so the lifter's `suffix == "H1"` check missed it and
        F2FP.UNPACK_B converted the LOW half (ex94 fp8 GEMM: 43% of the
        output wrong). Returns (had_reuse, remaining_suffix_or_None);
        ''/None pass through unchanged.
        """
        if not suffix:
            return False, suffix
        parts = suffix.split(".")
        if "reuse" not in parts:
            return False, suffix
        parts = [p for p in parts if p != "reuse"]
        return True, (".".join(parts) if parts else None)

    @classmethod
    def parse(cls, operand_content):
        operand_content = operand_content.lstrip()

        name = operand_content
        reg_name = None
        is_reg = False
        is_const_mem = False
        is_mem_addr = False
        is_immediate = False
        prefix = None
        suffix = None
        reuse = False
        offset_or_imm = None
        index_reg = None
        const_mem_bank = None
        desc_reg = None

        # For aggregator pass, to allow correct parsing
        # if c[0x1][0x104]:c[0x1][0x100] is given, only parse c[0x1][0x100]
        # Similarly -0x2:-0x1 -> -0x1
        # However, we want to keep register as R2:R1 as it differentiates from R1 or R2
        if ":" in operand_content and "0x" in operand_content:
            operand_content = operand_content.split(":")[1]

        if operand_content.startswith("-"):
            prefix = "-"
            operand_content = operand_content[1:]
        elif operand_content.startswith("!"):
            prefix = "!"
            operand_content = operand_content[1:]
        elif operand_content.startswith("~"):
            prefix = "~"
            operand_content = operand_content[1:]
        
        if operand_content.startswith("|"):
            if prefix is None:
                prefix = "|"
            else:
                prefix += "|"
            operand_content = operand_content.replace("|", "")

        if operand_content.startswith("c["):
            bank_end_idx = operand_content.find("]")
            const_mem_bank = int(operand_content[2:bank_end_idx], 16)
            operand_content = operand_content[bank_end_idx + 2:]
            offset_end_idx = operand_content.find("]")
            suffix = operand_content[offset_end_idx + 2 :]
            had_reuse, suffix = cls._extract_reuse(suffix)
            if had_reuse:
                reuse = True
            operand_content = operand_content[:offset_end_idx]
            is_const_mem = True

        # SM90 warpgroup descriptor: gdesc[UR8].tnspA.tnspB or gdesc[URZ]
        # Used by HGMMA/WGMMA for TMA-based operand sourcing.
        # Store the entire gdesc expression as name; mark as non-register symbol.
        if operand_content.startswith("gdesc["):
            first_close = operand_content.index("]")
            desc_reg = operand_content[6:first_close]
            gdesc_suffix = operand_content[first_close + 1:]
            return cls(
                name=name,
                reg=desc_reg,
                is_reg=True,
                is_mem_addr=False,
                is_const_mem=False,
                prefix=None,
                suffix=gdesc_suffix if gdesc_suffix else None,
                offset=None,
                immediate=None,
                index_reg=None,
                const_mem_bank=None,
                is_immediate=False,
                desc_reg=desc_reg,
            )

        # Blackwell tensor-memory instructions address tmem[URn+imm].
        # Keep the original address space in name, and expose its register
        # and displacement to the instruction-specific lifter.
        if operand_content.startswith("tmem[") and operand_content.endswith("]"):
            operand_content = operand_content[4:]

        # SM90 descriptor memory: desc[UR4][R2.64+0x10]
        # Extract descriptor register, then parse inner bracket as memory address.
        if operand_content.startswith("desc["):
            first_close = operand_content.index("]")
            desc_reg = operand_content[5:first_close]
            inner = operand_content[first_close + 1:]
            # inner is "[R2.64+0x10]" — strip brackets
            if inner.startswith("[") and inner.endswith("]"):
                inner = inner[1:-1]
            operand_content = inner
            is_mem_addr = True

        if operand_content.startswith("["):
            operand_content = operand_content[1:-1]
            is_mem_addr = True

        # # Normalize register-immediate subtraction (e.g., "R2-0x10") into "+-"
        # # so the generic '+' splitter can model it as a reg with an offset.
        # operand_content = re.sub(
        #     r"(?P<reg>(?:U?R\\d+(?:\\.[A-Za-z0-9]+)?|U?P\\d+|RZ|URZ))-(?P<imm>0x[0-9a-fA-F]+|\\d+)",
        #     r"\\g<reg>+-\\g<imm>",
        #     operand_content,
        # )

        # Scientific-notation floats like "1.84e+19" contain a "+" in the
        # exponent that the split("+") below would mis-split. Shield the
        # exponent sign with sentinels, split, then restore.
        protected_content = re.sub(
            r'(\d)e\+(\d)', r'\1eXPLUS\2',
            re.sub(r'(\d)e-(\d)', r'\1eXMINUS\2', operand_content)
        )
        sub_operands = [s.replace('eXPLUS', 'e+').replace('eXMINUS', 'e-')
                        for s in protected_content.split("+")]
        for sub_operand in sub_operands:
            # Suboperand is an immediate(offset)
            if "R" not in sub_operand and "P" not in sub_operand:
                if "0x" in sub_operand:
                    offset_or_imm = int(sub_operand, 16)
                else: # Try match as decimal
                    if "QNAN" in sub_operand:
                        offset_or_imm = float("nan")
                    else:
                        try:
                            offset_or_imm = float(sub_operand)
                        except ValueError:
                            pass
                        
                # special case: -c[0x0][0x160] instead of c[0x0][-0x160]
                if prefix == "-" and offset_or_imm is not None and not is_const_mem:
                    offset_or_imm = -offset_or_imm
                    prefix = None
                continue
            
            # Otherwise suboperand is a register
            is_reg = True
            
            # BRX R12 -0x420 ;
            # Strange syntax, should be R12.64+-0x420
            if "-0x" in sub_operand:
                offset_or_imm = -int(sub_operand.split("-")[1], 16)
                sub_operand = sub_operand.split("-")[0]
            if "+0x" in sub_operand:
                offset_or_imm = int(sub_operand.split("+")[1], 16)
                sub_operand = sub_operand.split("+")[0]
            
            # Match suffix(.reuse, .H1, .X4)
            if "." in sub_operand:
                sub_operand, suffix = sub_operand.split(".", 1)
                had_reuse, suffix = cls._extract_reuse(suffix)
                if had_reuse:
                    reuse = True

            # Match prefix(-, !, ||)
            if sub_operand.startswith("-"):
                prefix = "-"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("!"):
                prefix = "!"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("~"):
                prefix = "~"
                sub_operand = sub_operand[1:]
            elif sub_operand.startswith("|"):
                prefix = "|"
                sub_operand = sub_operand[1:-1]

            if reg_name is not None:
                index_reg = sub_operand
            else:
                reg_name = sub_operand

        if offset_or_imm is not None and not is_reg and not is_mem_addr and not is_const_mem:
            is_immediate = True

        if is_const_mem and is_reg and offset_or_imm is None:
            offset_or_imm = 0

        return cls(
            name=name,
            reg=reg_name,
            is_reg=is_reg,
            is_mem_addr=is_mem_addr,
            is_const_mem=is_const_mem,
            prefix=prefix,
            suffix=suffix,
            offset=offset_or_imm if not is_immediate else None,
            immediate=offset_or_imm if is_immediate else None,
            index_reg=index_reg,
            const_mem_bank=const_mem_bank,
            is_immediate=is_immediate,
            reuse=reuse,
            desc_reg=desc_reg,
        )

    @classmethod
    def from_immediate(cls, name, value):
        return cls(
            name=name,
            reg=None,
            is_reg=False,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=None,
            suffix=None,
            offset=None,
            immediate=value,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=True,
        )

    @classmethod
    def pack_half2_from_immediates(cls, f0, f1):
        h0 = struct.unpack('<H', struct.pack('<e', f0))[0]
        h1 = struct.unpack('<H', struct.pack('<e', f1))[0]
        return (h1 << 16) | h0

    @classmethod
    def from_half2_immediates(cls, f0, f1):
        packed = cls.pack_half2_from_immediates(f0, f1)
        return cls.from_immediate(f"0x{packed:08x}", packed)

    @classmethod
    def from_cm(cls, name, arg_offset, reg_name=None, prefix=None):
        return cls(
            name=name,
            reg=reg_name,
            is_reg=False,
            is_mem_addr=False,
            is_const_mem=True,
            prefix=prefix,
            suffix=None,
            offset=arg_offset,
            immediate=None,
            index_reg=None,
            const_mem_bank=0,
            is_immediate=False,
        )

    @classmethod
    def from_mem_addr(cls, name, reg, suffix=None, offset=0, index_reg=None, reuse=False):
        return cls(
            name=name,
            reg=reg,
            is_reg=True,
            is_mem_addr=True,
            is_const_mem=False,
            prefix=None,
            suffix=suffix,
            offset=offset,
            immediate=None,
            index_reg=index_reg,
            const_mem_bank=None,
            is_immediate=False,
            reuse=reuse,
        )
        
    @classmethod
    def from_reg(cls, name, reg, suffix=None, prefix=None, reuse=False):
        return cls(
            name=name,
            reg=reg,
            is_reg=True,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=prefix,
            suffix=suffix,
            offset=None,
            immediate=None,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=False,
            reuse=reuse,
        )

    @classmethod
    def from_symbol(cls, name):
        return cls(
            name=name,
            reg=None,
            is_reg=False,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=None,
            suffix=None,
            offset=None,
            immediate=None,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=False,
        )

    @classmethod
    def create_reg_group(cls, operand, count):
        if not operand.is_reg:
            return operand.clone()

        if operand.is_reg_group:
            return operand.clone()

        base_reg = operand.reg

        if "RZ" in base_reg or "PT" in base_reg:
            regs = [base_reg] * count
        else:
            is_ur = "UR" in base_reg
            prefix_len = 2 if is_ur else 1
            prefix_str = "UR" if is_ur else "R"
            base_num = int(base_reg[prefix_len:])
            regs = [f"{prefix_str}{base_num + i}" for i in range(count)]

        reg_ops = [cls.from_reg(r, r) for r in regs]
        group_name = ":".join(reversed(regs))

        return cls(
            name=group_name,
            reg=None,
            is_reg=True,
            is_mem_addr=operand.is_mem_addr,
            is_const_mem=False,
            prefix=operand.prefix,
            suffix=operand.suffix,
            offset=operand.offset_value,
            immediate=None,
            # Preserve the uniform/index register of reg+reg addressing
            # ([R16.64+UR4]): dropping it here silently loses the +UR4 term
            # (OpModTransform materializes it into an explicit IADD64).
            index_reg=operand.index_reg,
            const_mem_bank=None,
            is_immediate=False,
            reg_group=reg_ops,
            reuse=operand.reuse,
            desc_reg=getattr(operand, 'desc_reg', None),
        )

    @classmethod
    def create_reg_pair(cls, operand):
        return cls.create_reg_group(operand, 2)

    @classmethod
    def create_reg_quad(cls, operand):
        return cls.create_reg_group(operand, 4)

    @classmethod
    def from_reg_group(cls, name, regs):
        reg_ops = [cls.from_reg(r, r) for r in regs]
        return cls(
            name=name,
            reg=None,
            is_reg=True,
            is_mem_addr=False,
            is_const_mem=False,
            prefix=None,
            suffix=None,
            offset=None,
            immediate=None,
            index_reg=None,
            const_mem_bank=None,
            is_immediate=False,
            reg_group=reg_ops,
        )

    @classmethod
    def from_reg_pair(cls, name, reg_lower, reg_upper):
        """Create a 64-bit register pair from two register names."""
        return cls.from_reg_group(name, [reg_lower, reg_upper])

    @classmethod
    def from_reg_quad(cls, name, reg0, reg1, reg2, reg3):
        """Create a 128-bit register quad from four register names."""
        return cls.from_reg_group(name, [reg0, reg1, reg2, reg3])

    def __init__(
        self,
        name,
        reg,
        is_reg,
        is_mem_addr,
        is_const_mem,
        is_immediate=False,
        prefix=None,
        suffix=None,
        offset=None,
        immediate=None,
        index_reg=None,
        const_mem_bank=None,
        reg_group=None,
        reuse=False,
        desc_reg=None,
    ):
        self.name = name
        self.reg = reg
        self.reg_group = reg_group
        self.is_reg = is_reg
        self.is_mem_addr = is_mem_addr
        self.is_const_mem = is_const_mem
        self.is_immediate = is_immediate or immediate is not None
        self.prefix = prefix
        self.suffix = suffix
        self.reuse = reuse
        self.offset_value = offset
        self.immediate_value = immediate if self.is_immediate else None
        self.index_reg = index_reg
        self.const_mem_bank = const_mem_bank
        self.desc_reg = desc_reg
        self.type_desc = "NOTYPE"
        self.ir_type = None
        self.ir_reg_name = None
        self.defining_insts = set()
        self.implicit = False

    @property
    def is_reg_group(self):
        # Canonical predicate for "this operand is a register group."
        # Use this for conditionals; use the `reg_group` attribute only
        # when you need to read or iterate the list itself.
        return self.reg_group is not None and len(self.reg_group) > 0

    @property
    def is_reg_pair(self):
        return self.reg_group is not None and len(self.reg_group) == 2

    @property
    def is_reg_quad(self):
        return self.reg_group is not None and len(self.reg_group) == 4

    @property
    def is_arg(self):
        return self.is_const_mem and self.const_mem_bank == 0

    @property
    def offset(self):
        if self.is_mem_addr or self.is_const_mem:
            return self.offset_value
        return None

    @property
    def is_float_immediate(self):
        return self.is_immediate and self.name and "0x" not in self.name.lower()
    
    @property
    def is_negative_reg(self):
        return self.prefix and "-" in self.prefix

    @property
    def is_not_reg(self):
        return self.prefix and any(p in self.prefix for p in ("!", "~"))

    @property
    def is_abs_reg(self):
        return self.prefix and "|" in self.prefix

    @property
    def is_predicate_reg(self):
        if self.is_reg_group:
            op = self.reg_group[0]
        else:
            op = self

        if not op.is_reg:
            return False

        if "P" not in op.reg:
            return False

        reg = op.reg
        reg = reg[1:] if reg.startswith("U") else reg
        reg = reg[1:] if reg.startswith("P") else reg

        return reg[0].isdigit()

    @property
    def is_uniform_reg(self):
        if not self.is_reg:
            return False

        return self.reg.startswith("U")

    @property
    def is_writable_reg(self):
        # Hot path (millions of calls per lift): inlined is_pt/is_rz/
        # is_special_reg/is_barrier_reg checks, semantics identical.
        if not self.is_reg:
            return False
        if self.name and self.name.startswith(_SPECIAL_REG_PREFIXES):
            return False
        reg = self.reg
        if reg:
            if reg[0] == "B":
                return False
            base = reg.split("@", 1)[0] if "@" in reg else reg
            if base in ("RZ", "SRZ", "URZ", "PT", "UPT"):
                return False
        return True

    @property
    def is_barrier_reg(self):
        return self.reg and self.reg[0] == "B"

    @property
    def is_special_reg(self):
        return bool(self.name and self.name.startswith(_SPECIAL_REG_PREFIXES))

    @property
    def is_thread_idx(self):
        return self.name and self.name.startswith(SR_TID)

    @property
    def is_thread_idx_x(self):
        return self.is_thread_idx and "X" in self.name

    @property
    def is_thread_idx_y(self):
        return self.is_thread_idx and "Y" in self.name

    @property
    def is_thread_idx_z(self):
        return self.is_thread_idx and "Z" in self.name

    @property
    def is_block_dim(self):
        return self.name and self.name.startswith(SR_NTID)

    @property
    def is_block_dim_x(self):
        return self.is_block_dim and "X" in self.name

    @property
    def is_block_dim_y(self):
        return self.is_block_dim and "Y" in self.name

    @property
    def is_block_dim_z(self):
        return self.is_block_dim and "Z" in self.name
    
    @property
    def is_block_idx(self):
        return self.name and self.name.startswith(SR_CTAID)

    @property
    def is_block_idx_x(self):
        return self.is_block_idx and "X" in self.name

    @property
    def is_block_idx_y(self):
        return self.is_block_idx and "Y" in self.name
    
    @property
    def is_block_idx_z(self):
        return self.is_block_idx and "Z" in self.name

    @property
    def is_grid_dim(self):
        return self.name and self.name.startswith(SR_GRID_DIM)

    @property
    def is_grid_dim_x(self):
        return self.is_grid_dim and "X" in self.name

    @property
    def is_grid_dim_y(self):
        return self.is_grid_dim and "Y" in self.name

    @property
    def is_grid_dim_z(self):
        return self.is_grid_dim and "Z" in self.name

    @property
    def is_lane_id(self):
        return self.name and self.name.startswith(SR_LANE)

    @property
    def is_warp_id(self):
        return self.name and self.name.startswith(SR_WARP)

    @property
    def is_warp_size(self):
        return self.name and (
            self.name.startswith(SR_WARPSIZE) or self.name.startswith(SR_WARPSZ)
        )

    IsWarpSize = is_warp_size

    @property
    def is_lane_mask_eq(self):
        return self.name and self.name.startswith(SR_EQMASK)
    
    @property
    def is_lane_mask_le(self):
        return self.name and self.name.startswith(SR_LEMASK)
    
    @property
    def is_lane_mask_lt(self):
        return self.name and self.name.startswith(SR_LTMASK)

    @property
    def is_lane_mask_ge(self):
        return self.name and self.name.startswith(SR_GEMASK)

    @property
    def is_lane_mask_gt(self):
        return self.name and self.name.startswith(SR_GTMASK)

    @property
    def is_active_mask(self):
        return self.name and self.name.startswith(SR_ACTIVEMASK)

    @property
    def is_cluster_ctarank(self):
        # SM90 SR_CgaCtaId: the CTA's linear rank within its cluster (CGA).
        # PTX equivalent: %cluster_ctarank. ptxas reads it for shared::cluster
        # window addressing ((rank << 24) | offset) even in kernels that never
        # use clusters in source (rank is 0 at runtime for non-cluster
        # launches) — e.g. plain __shared__ reductions like lda-cuda.
        return bool(self.name and self.name.startswith(SR_CGACTAID))

    @property
    def is_clock_lo(self):
        return self.name and (self.name == "SR_CLOCKLO" or self.name == "SR_CLOCK")

    @property
    def is_clock_hi(self):
        return self.name and self.name == "SR_CLOCKHI"

    @property
    def is_global_timer_lo(self):
        return self.name and self.name == "SR_GLOBALTIMERLO"

    @property
    def is_global_timer_hi(self):
        return self.name and self.name == "SR_GLOBALTIMERHI"

    @property
    def is_rz(self):
        if not self.reg:
            return False
        # Strip SSA suffix (e.g., RZ@0 -> RZ) before checking
        base_reg = self.reg.split("@")[0] if "@" in self.reg else self.reg
        return base_reg in ["RZ", "SRZ", "URZ"]
    
    @property
    def is_pt(self):
        if not self.reg:
            return False
        # Strip SSA suffix (e.g., PT@2 -> PT) before checking
        base_reg = self.reg.split("@")[0] if "@" in self.reg else self.reg
        return base_reg in ["PT", "UPT"]

    def __str__(self):
        base_str = ""

        if self.is_const_mem:
            parts = []
            if self.reg:
                parts.append(self.reg)
            
            if self.offset_value is not None:
                offset = int(self.offset_value)
                if offset != 0 or not parts:
                    if offset < 0:
                        parts.append(f"-0x{abs(offset):x}")
                    elif parts:
                        parts.append(f"+0x{offset:x}")
                    else:
                        parts.append(f"0x{offset:x}")
            
            base_str = f"c[0x{self.const_mem_bank:x}][{''.join(parts)}]"
        elif self.is_immediate:
            if self.is_float_immediate:
                import math
                v = self.immediate_value
                # Some "float" immediates carry a wide bit pattern in
                # an int field (e.g. parser stored an i64 raw bits as
                # the immediate value). math.isinf/isnan only accept
                # floats; converting an int wider than ~1.8e308 raises
                # OverflowError. Convert defensively.
                try:
                    fv = float(v)
                except (OverflowError, TypeError, ValueError):
                    base_str = hex(int(v)) if isinstance(v, int) else str(v)
                else:
                    if math.isinf(fv):
                        base_str = "+INF" if fv > 0 else "-INF"
                    elif math.isnan(fv):
                        base_str = "-QNAN"
                    else:
                        base_str = str(v)
            else:
                base_str = hex(self.immediate_value)
        elif self.is_mem_addr:
            parts = []

            core_name = ""
            if self.is_reg_group and self.reg_group[0].reg:
                core_name = self.reg_group[0].reg
            elif self.reg:
                core_name = self.reg
            elif self.name:
                if self.name.startswith("[") and self.name.endswith("]"):
                    core_name = self.name[1:-1]
                else:
                    core_name = self.name

            if core_name:
                if self.suffix:
                    if self.suffix == "reuse":
                        core_name = f"{core_name}.reuse"
                    else:
                        core_name = f"{core_name}.{self.suffix}"
                if self.reuse and self.suffix != "reuse":
                    core_name = f"{core_name}.reuse"
                parts.append(core_name)

            if self.index_reg:
                parts.append(f"+{self.index_reg}")

            if self.offset_value is not None:
                offset = int(self.offset_value)
                if offset != 0 or not parts:
                    if offset < 0:
                        parts.append(f"-0x{abs(offset):x}")
                    else:
                        prefix_sign = "+" if parts else ""
                        parts.append(f"{prefix_sign}0x{offset:x}")

            base_str = f"[{''.join(parts)}]"
        elif self.is_reg_group:
            base_str = ":".join(str(op) for op in reversed(self.reg_group))
            if self.offset_value is not None:
                offset = int(self.offset_value)
                if offset < 0:
                    base_str += f"-0x{abs(offset):x}"
                elif offset > 0:
                    base_str += f"+0x{offset:x}"
        elif self.reg:
            base_str = self.reg
            if self.offset_value is not None:
                offset = int(self.offset_value)
                if offset < 0:
                    base_str += f"-0x{abs(offset):x}"
                elif offset > 0:
                    base_str += f"+0x{offset:x}"
        elif self.name:
            base_str = self.name
        else:
            base_str = "<??>"

        if self.suffix or self.reuse:
            if not self.is_mem_addr:
                if self.suffix:
                    if self.suffix == "reuse":
                        base_str = f"{base_str}.reuse"
                    else:
                        base_str = f"{base_str}.{self.suffix}"
                if self.reuse and self.suffix != "reuse":
                    base_str = f"{base_str}.reuse"

        if self.prefix:
            s = base_str
            if "|" in self.prefix:
                s = f"|{s}|"
            
            # Handle other prefixes
            other_prefix = self.prefix.replace("|", "")
            if other_prefix:
                s = f"{other_prefix}{s}"
            
            return s

        return base_str

    def __repr__(self):
        return self.__str__()

    def set_reg(self, reg_name):
        self.reg = reg_name
        self.name = None
        self.is_reg = True
        self.is_immediate = False
        self.name = str(self)

    def replace(self, other):
        self.name = other.name
        self.reg = other.reg
        self.reg_group = other.reg_group
        self.is_reg = other.is_reg
        self.is_mem_addr = other.is_mem_addr
        self.is_const_mem = other.is_const_mem
        self.prefix = other.prefix
        self.suffix = other.suffix
        self.reuse = other.reuse
        self.offset_value = other.offset_value
        self.immediate_value = other.immediate_value
        self.index_reg = other.index_reg
        self.const_mem_bank = other.const_mem_bank
        self.is_immediate = other.is_immediate
        self.type_desc = other.type_desc
        self.ir_type = other.ir_type
        self.ir_reg_name = other.ir_reg_name
        self.defining_insts = other.defining_insts

    def set_type_desc(self, ty):
        self.type_desc = ty

    def get_type_desc(self):
        return self.type_desc

    def has_type_desc(self):
        return self.type_desc != "NOTYPE"

    def get_ir_type(self, lifter):
        return lifter.get_ir_type(self.type_desc)

    def get_ir_name(self, lifter):
        if self.is_reg_group:
            return ":".join(op.reg for op in reversed(self.reg_group))
        if self.is_reg:
            return self.reg
        if self.is_const_mem:
            return f"c[0x0{self.const_mem_bank:x}][0x{self.offset_value:x}]"
        return None

    def dump(self):
        print("operand: ", self.name, self.reg)

    def clone(self):
        cloned = Operand(
            name=self.name,
            reg=self.reg,
            is_reg=self.is_reg,
            is_mem_addr=self.is_mem_addr,
            is_const_mem=self.is_const_mem,
            prefix=self.prefix,
            suffix=self.suffix,
            offset=self.offset_value,
            immediate=self.immediate_value,
            index_reg=self.index_reg,
            const_mem_bank=self.const_mem_bank,
            is_immediate=self.is_immediate,
            reg_group=[op.clone() for op in self.reg_group] if self.is_reg_group else None,
            reuse=self.reuse,
            desc_reg=self.desc_reg,
        )
        cloned.type_desc = self.type_desc
        cloned.ir_type = self.ir_type
        cloned.ir_reg_name = self.ir_reg_name
        cloned.defining_insts = set(self.defining_insts)
        for name in ("desc_hi", "desc_a_hi", "desc_b_hi", "wgmma_a_frag", "wgmma_neg_a",
                     "wgmma_sparse_meta", "wgmma_sparse_selector"):
            if hasattr(self, name):
                setattr(cloned, name, getattr(self, name))
        return cloned
