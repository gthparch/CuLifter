"""DWARF debug info parser for CUDA cubins.

Extracts ground-truth type information from debug cubins compiled with
nvcc -G -g. Uses pyelftools to parse DWARF DIEs and map source-level
types to SLifter's type lattice.

Usage:
    from utils.dwarf import parse_dwarf_types
    ground_truth = parse_dwarf_types("kernel_debug.cubin")
"""

from dataclasses import dataclass, field
from pathlib import Path
from elftools.elf.elffile import ELFFile


# ---------------------------------------------------------------------------
# Data classes for ground-truth representation
# ---------------------------------------------------------------------------

@dataclass
class ParamGT:
    """Ground-truth type for a kernel parameter."""
    ordinal: int          # parameter index (0-based)
    name: str             # source name (e.g., "A")
    c_type: str           # source type string (e.g., "float *")
    slifter_type: str     # mapped SLifter type (e.g., "Int64" for pointers)
    size: int             # size in bytes
    c0_offset: int = -1   # constant memory offset (populated by ABI mapping)


@dataclass
class LocalVarGT:
    """Ground-truth type for a local variable."""
    name: str
    c_type: str
    slifter_type: str
    size: int
    register: str = ""    # SASS register (e.g., "R5"), if extractable from DWARF location


@dataclass
class KernelGT:
    """Ground-truth types for a single kernel function."""
    name: str
    params: list = field(default_factory=list)   # list[ParamGT]
    locals_: list = field(default_factory=list)   # list[LocalVarGT]


# ---------------------------------------------------------------------------
# DWARF base type -> SLifter type mapping
# ---------------------------------------------------------------------------

def _dwarf_to_slifter_type(die, dwarf_info):
    """Resolve a DWARF type DIE to a (c_type_string, slifter_type, size) tuple.

    Chases DW_AT_type references through typedefs, const qualifiers, and
    pointer types until a base type or pointer is reached.
    """
    if die is None:
        return ("void", "Int32", 0)

    visited = set()
    current = die
    pointer_depth = 0

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        tag = current.tag

        if tag == "DW_TAG_pointer_type":
            pointer_depth += 1
            if "DW_AT_type" in current.attributes:
                current = _resolve_type_ref(current, dwarf_info)
            else:
                # void pointer
                return ("void *" + " *" * (pointer_depth - 1), "Int64", 8)

        elif tag in ("DW_TAG_typedef", "DW_TAG_const_type",
                      "DW_TAG_volatile_type", "DW_TAG_restrict_type"):
            if "DW_AT_type" in current.attributes:
                current = _resolve_type_ref(current, dwarf_info)
            else:
                return ("void", "Int32", 0)

        elif tag == "DW_TAG_base_type":
            name = _attr_str(current, "DW_AT_name")
            size = _attr_int(current, "DW_AT_byte_size", 4)
            encoding = _attr_int(current, "DW_AT_encoding", 0)

            if pointer_depth > 0:
                return (name + " *" * pointer_depth, "Int64", 8)

            slifter_type = _base_type_to_slifter(name, size, encoding)
            return (name, slifter_type, size)

        elif tag == "DW_TAG_structure_type":
            name = _attr_str(current, "DW_AT_name") or "<struct>"
            size = _attr_int(current, "DW_AT_byte_size", 0)
            if pointer_depth > 0:
                return (name + " *" * pointer_depth, "Int64", 8)
            # Struct passed by value — map to Int based on size
            slifter_type = f"Int{size * 8}" if size in (4, 8, 16) else "Int32"
            return (name, slifter_type, size)

        elif tag == "DW_TAG_array_type":
            if pointer_depth > 0:
                return ("array *" * pointer_depth, "Int64", 8)
            current = _resolve_type_ref(current, dwarf_info)

        elif tag == "DW_TAG_enumeration_type":
            size = _attr_int(current, "DW_AT_byte_size", 4)
            if pointer_depth > 0:
                return ("enum *" * pointer_depth, "Int64", 8)
            return ("enum", "Int32", size)

        else:
            # Unknown tag — bail out
            break

    return ("unknown", "Int32", 4)


def _base_type_to_slifter(name, size, encoding):
    """Map a DWARF base type to SLifter type string."""
    # DW_ATE encoding constants
    DW_ATE_float = 0x04
    DW_ATE_signed = 0x05
    DW_ATE_unsigned = 0x07
    DW_ATE_boolean = 0x02
    DW_ATE_unsigned_char = 0x08
    DW_ATE_signed_char = 0x06

    if encoding == DW_ATE_boolean:
        return "Bool"
    elif encoding == DW_ATE_float:
        if size == 2:
            return "Float16"
        elif size == 4:
            return "Float32"
        elif size == 8:
            return "Float64"
        return "Float32"
    elif encoding in (DW_ATE_signed, DW_ATE_unsigned,
                      DW_ATE_signed_char, DW_ATE_unsigned_char):
        if size <= 4:
            return "Int32"
        elif size == 8:
            return "Int64"
        elif size == 16:
            return "Int128"
        return "Int32"

    # Fallback: guess from name
    name_lower = (name or "").lower()
    if "float" in name_lower:
        return "Float64" if size == 8 else "Float32"
    if "double" in name_lower:
        return "Float64"
    if "half" in name_lower or "__half" in name_lower:
        return "Float16"
    if "bool" in name_lower:
        return "Bool"
    if size == 8:
        return "Int64"
    return "Int32"


def _resolve_type_ref(die, dwarf_info):
    """Follow DW_AT_type reference to the target DIE."""
    if "DW_AT_type" not in die.attributes:
        return None
    ref = die.attributes["DW_AT_type"]
    offset = ref.value
    # In pyelftools, ref.value is a CU-relative offset for DW_FORM_ref*
    # or a global offset for DW_FORM_ref_addr.
    form = ref.form
    if form.startswith("DW_FORM_ref") and form != "DW_FORM_ref_addr":
        # CU-relative offset
        cu = die.cu
        offset = cu.cu_offset + offset
    try:
        return dwarf_info.get_DIE_from_refaddr(offset)
    except Exception:
        return None


def _attr_str(die, attr_name):
    """Get a string attribute from a DIE, or empty string."""
    if attr_name not in die.attributes:
        return ""
    val = die.attributes[attr_name].value
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="replace")
    return str(val)


def _attr_int(die, attr_name, default=0):
    """Get an integer attribute from a DIE, or default."""
    if attr_name not in die.attributes:
        return default
    return die.attributes[attr_name].value


def _extract_register_from_location(die):
    """Try to extract a SASS register name from DW_AT_location.

    DWARF location expressions for GPU registers use DW_OP_regN or
    DW_OP_regx with a register number. We map these to SASS register
    names (R0, R1, ...).

    Returns register name string or empty string if not extractable.
    """
    if "DW_AT_location" not in die.attributes:
        return ""

    loc_attr = die.attributes["DW_AT_location"]

    # Simple location expression (not a location list)
    if hasattr(loc_attr, "value") and isinstance(loc_attr.value, list):
        ops = loc_attr.value
        if len(ops) >= 1:
            op = ops[0]
            # DW_OP_reg0..DW_OP_reg31 = opcodes 0x50..0x6f
            if hasattr(op, "op") and 0x50 <= op.op <= 0x6f:
                reg_num = op.op - 0x50
                return f"R{reg_num}"
            # DW_OP_regx = opcode 0x90 with ULEB128 operand
            if hasattr(op, "op") and op.op == 0x90:
                if hasattr(op, "args") and op.args:
                    return f"R{op.args[0]}"
    return ""


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------

def parse_dwarf_types(cubin_path):
    """Parse DWARF from a debug cubin and extract ground-truth types.

    Args:
        cubin_path: Path to a cubin compiled with nvcc -G -g.

    Returns:
        dict mapping kernel name -> KernelGT
    """
    cubin_path = Path(cubin_path)
    results = {}

    with open(cubin_path, "rb") as f:
        try:
            elffile = ELFFile(f)
        except Exception as e:
            print(f"Warning: Cannot parse ELF from {cubin_path}: {e}")
            return results

        if not elffile.has_dwarf_info():
            print(f"Warning: No DWARF info in {cubin_path}")
            return results

        # CUDA debug cubins carry relocations against the DWARF sections that
        # pyelftools mis-handles (it looks up a symtab section that lacks
        # num_symbols), crashing get_dwarf_info(). We only need the type DIEs
        # (names, encodings, sizes, type refs), not relocated addresses, so we
        # skip relocation application.
        dwarf_info = elffile.get_dwarf_info(relocate_dwarf_sections=False)

        for cu in dwarf_info.iter_CUs():
            top_die = cu.get_top_DIE()
            _walk_die_for_functions(top_die, dwarf_info, results)

    return results


def _walk_die_for_functions(die, dwarf_info, results):
    """Recursively walk DIEs to find DW_TAG_subprogram (functions)."""
    if die.tag == "DW_TAG_subprogram":
        name = _attr_str(die, "DW_AT_name")
        if name:
            kernel_gt = _extract_function_types(die, dwarf_info, name)
            results[name] = kernel_gt

    for child in die.iter_children():
        _walk_die_for_functions(child, dwarf_info, results)


def _extract_function_types(func_die, dwarf_info, func_name):
    """Extract parameter and local variable types from a function DIE."""
    kernel_gt = KernelGT(name=func_name)
    param_ordinal = 0

    for child in func_die.iter_children():
        if child.tag == "DW_TAG_formal_parameter":
            param_name = _attr_str(child, "DW_AT_name")
            c_type, slifter_type, size = _dwarf_to_slifter_type(
                _resolve_type_ref(child, dwarf_info), dwarf_info
            )
            param = ParamGT(
                ordinal=param_ordinal,
                name=param_name,
                c_type=c_type,
                slifter_type=slifter_type,
                size=size,
            )
            kernel_gt.params.append(param)
            param_ordinal += 1

        elif child.tag == "DW_TAG_variable":
            var_name = _attr_str(child, "DW_AT_name")
            c_type, slifter_type, size = _dwarf_to_slifter_type(
                _resolve_type_ref(child, dwarf_info), dwarf_info
            )
            register = _extract_register_from_location(child)
            local_var = LocalVarGT(
                name=var_name,
                c_type=c_type,
                slifter_type=slifter_type,
                size=size,
                register=register,
            )
            kernel_gt.locals_.append(local_var)

    return kernel_gt


# ---------------------------------------------------------------------------
# Source-derived ground truth (Track 2: parse .cu kernel signatures)
# ---------------------------------------------------------------------------

import re

# Regex for CUDA kernel signatures
_KERNEL_SIG_RE = re.compile(
    r'(?:extern\s+"C"\s+)?'
    r"__global__\s+void\s+"
    r"(\w+)\s*\("
    r"([^)]*)"
    r"\)",
    re.MULTILINE,
)

# Map C type keywords to SLifter types
_C_TYPE_MAP = {
    "float": ("Float32", 4),
    "double": ("Float64", 8),
    "int": ("Int32", 4),
    "unsigned int": ("Int32", 4),
    "unsigned": ("Int32", 4),
    "long long": ("Int64", 8),
    "unsigned long long": ("Int64", 8),
    "int64_t": ("Int64", 8),
    "uint64_t": ("Int64", 8),
    "int32_t": ("Int32", 4),
    "uint32_t": ("Int32", 4),
    "size_t": ("Int64", 8),
    "bool": ("Bool", 1),
    "char": ("Int32", 1),
    "unsigned char": ("Int32", 1),
    "short": ("Int32", 2),
    "unsigned short": ("Int32", 2),
    "__half": ("Float16", 2),
    "half": ("Float16", 2),
}


def parse_source_types(cu_path, sm=75):
    """Parse kernel signatures from a .cu source file.

    Args:
        cu_path: Path to .cu source file.
        sm: SM version (affects ABI parameter base offset and alignment).
            SM75: base=0x160, align=max(size,4)
            SM90: base=0x208, align=8 (all params 8-byte aligned)

    Returns dict mapping kernel name -> KernelGT (params only, no locals).
    """
    cu_path = Path(cu_path)
    text = cu_path.read_text(encoding="utf-8", errors="replace")
    results = {}

    # SM-dependent ABI constants
    # SM90: params start at c[0][0x210] (0x208 is the TMA descriptor base),
    #        all params occupy 8-byte aligned slots.
    # SM75: params start at c[0][0x160], natural alignment (min 4 bytes).
    if sm >= 90:
        param_base = 0x210
        min_align = 8  # SM90: all params in 8-byte slots
    else:
        param_base = 0x160
        min_align = 4  # SM75: natural alignment, min 4 bytes

    for match in _KERNEL_SIG_RE.finditer(text):
        func_name = match.group(1)
        params_str = match.group(2).strip()
        if not params_str:
            results[func_name] = KernelGT(name=func_name)
            continue

        kernel_gt = KernelGT(name=func_name)
        c0_offset = param_base

        for ordinal, param in enumerate(params_str.split(",")):
            param = param.strip()
            if not param:
                continue

            is_pointer = "*" in param
            # Remove qualifiers
            clean = param
            for qual in ("const", "__restrict__", "volatile", "extern"):
                clean = clean.replace(qual, "")
            clean = clean.strip()

            # Split into type and name
            parts = clean.rsplit(None, 1)
            if len(parts) == 2:
                type_str, name = parts
            else:
                type_str, name = parts[0], f"arg{ordinal}"

            type_str = type_str.replace("*", "").strip()

            if is_pointer:
                slifter_type = "Int64"
                size = 8
            else:
                slifter_type, size = _C_TYPE_MAP.get(type_str, ("Int32", 4))

            c_type_display = f"{type_str} *" if is_pointer else type_str

            # Align offset for this parameter
            align = max(size, min_align)
            c0_offset = ((c0_offset + align - 1) // align) * align

            p = ParamGT(
                ordinal=ordinal,
                name=name,
                c_type=c_type_display,
                slifter_type=slifter_type,
                size=size,
                c0_offset=c0_offset,
            )
            kernel_gt.params.append(p)

            # Advance past this parameter (SM90: always 8-byte slots)
            c0_offset += max(size, min_align)

        results[func_name] = kernel_gt

    return results


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

import json


def ground_truth_to_json(gt_dict):
    """Convert ground truth dict to JSON-serializable dict."""
    out = {}
    for name, kgt in gt_dict.items():
        out[name] = {
            "name": kgt.name,
            "params": [
                {
                    "ordinal": p.ordinal,
                    "name": p.name,
                    "c_type": p.c_type,
                    "slifter_type": p.slifter_type,
                    "size": p.size,
                    "c0_offset": p.c0_offset,
                }
                for p in kgt.params
            ],
            "locals": [
                {
                    "name": v.name,
                    "c_type": v.c_type,
                    "slifter_type": v.slifter_type,
                    "size": v.size,
                    "register": v.register,
                }
                for v in kgt.locals_
            ],
        }
    return out


def ground_truth_from_json(data):
    """Load ground truth dict from JSON-serializable dict."""
    out = {}
    for name, kdata in data.items():
        kgt = KernelGT(name=kdata["name"])
        for p in kdata.get("params", []):
            kgt.params.append(ParamGT(**p))
        for v in kdata.get("locals", []):
            kgt.locals_.append(LocalVarGT(**v))
        out[name] = kgt
    return out


def save_ground_truth(gt_dict, output_path):
    """Save ground truth to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(ground_truth_to_json(gt_dict), f, indent=2)


def load_ground_truth(json_path):
    """Load ground truth from a JSON file."""
    with open(json_path) as f:
        return ground_truth_from_json(json.load(f))
