#!/usr/bin/env python3
"""Analyze type conflicts in detail — log type pairs, opcodes, context.

Auto-discovers all SM90 kernels in test/cuobjdump/ and test/addrtest/,
classifies each into a domain, and reports conflict patterns.

Usage:
    export PATH="$PWD/.conda_env/bin:$PATH"
    python -m reproduce.figures.engines.analyze_conflicts.py
"""
import sys
import os
import json
import re
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from parse.parser import SaSSParser
from sir.module import Module
from lift.lifter_cpu import CpuLifter as X86Lifter
from utils.elf import Elf


@dataclass
class ConflictRecord:
    reg: str
    prev_type: str       # type set before conflict (e.g. "Float32")
    new_type: str        # type set from conflicting instruction
    def_opcode: str      # opcode that defined this register's type
    use_opcode: str      # opcode that caused the conflict
    use_inst_str: str    # full instruction string at conflict site
    def_inst_str: str    # full instruction string at definition site (if known)


def format_typeset(ts):
    """Convert a frozenset to a readable sorted string."""
    if not ts:
        return "{}"
    return "{" + ", ".join(sorted(ts)) + "}"


def lift_and_collect_conflicts(sass_path, elf_path):
    """Run the pipeline up through TypeAnalysis, collecting detailed conflicts."""

    sass_text = Path(sass_path).read_text()
    elf = Elf(str(elf_path))
    parser = SaSSParser(sass_text, elf=elf)
    module = Module("analysis", parser)

    lifter = X86Lifter(elf=elf, verbose=False)
    passes = lifter.get_transform_passes()

    conflicts = []

    for p in passes:
        p.apply(module)

        # After TypeAnalysis runs, extract conflict details
        if type(p).__name__ == "TypeAnalysis":
            # Collect from patched data (see monkey-patch below)
            if hasattr(p, '_conflict_log'):
                conflicts = p._conflict_log

    return conflicts, module


def patch_typeanalysis_for_logging():
    """Monkey-patch TypeAnalysis.set_op_type to log detailed conflict info."""
    from transform.typeanalysis import TypeAnalysis

    original_set_op_type = TypeAnalysis.set_op_type

    def logging_set_op_type(self, optypes, operand, type_set, inst):
        if not hasattr(self, '_conflict_log'):
            self._conflict_log = []
        if not hasattr(self, '_reg_def_inst'):
            self._reg_def_inst = {}

        previous_type = self.get_optype(optypes, operand)

        # Track which instruction first seeded this register
        reg_key = operand.reg if hasattr(operand, 'reg') else str(operand)
        if reg_key not in self._reg_def_inst and previous_type != type_set:
            self._reg_def_inst[reg_key] = (format_typeset(previous_type),
                                           inst.opcodes[0] if inst.opcodes else "?",
                                           str(inst).strip()[:120])

        # Detect conflict
        if len(previous_type & type_set) == 0 and reg_key not in self.conflicts:
            def_info = self._reg_def_inst.get(reg_key, (format_typeset(previous_type), "?", "?"))

            record = ConflictRecord(
                reg=reg_key,
                prev_type=format_typeset(previous_type),
                new_type=format_typeset(type_set),
                def_opcode=def_info[1],
                use_opcode=inst.opcodes[0] if inst.opcodes else "?",
                use_inst_str=str(inst).strip()[:120],
                def_inst_str=def_info[2],
            )
            self._conflict_log.append(record)

        # Call original (MUST return its result: set_op_type recurses on vector
        # sub-operands via `changed |= self.set_op_type(...)`, so dropping the
        # return value yields `bool |= None` -> TypeError on any vector operand).
        return original_set_op_type(self, optypes, operand, type_set, inst)

    TypeAnalysis.set_op_type = logging_set_op_type


def classify_conflict(c):
    """Attempt to classify a conflict into a known pattern."""
    prev = c.prev_type
    new = c.new_type

    # ── Float32 → Int32 conflicts ──────────────────────────────────
    if "Float32" in prev and "Int32" in new:
        pair = "Float32→Int32"
        # Exponent extraction: SHF by 23 on float (IEEE 754 exponent field)
        if c.use_opcode == "SHF" and "0x17" in c.use_inst_str:
            return pair, "exponent_extraction"
        # Other shifts on float bits (sign extraction, etc.)
        if c.use_opcode == "SHF":
            return pair, "bit_field_extraction"
        # rsqrt range check: IADD3 with magic constants on float bits
        if c.use_opcode == "IADD3" and ("0xd000000" in c.use_inst_str or
                                         "0x5f3759df" in c.use_inst_str):
            return pair, "rsqrt_range_check"
        # Integer arithmetic on float bits (exponent bias, etc.)
        if c.use_opcode in ("IADD3", "IADD32I", "IADD", "IMAD"):
            # Check context for MUFU nearby (Newton-Raphson refinement)
            return pair, "int_arith_on_float_bits"
        # Mantissa/sign extraction via LOP3 with IEEE masks
        if c.use_opcode == "LOP3":
            use_str = c.use_inst_str
            if "0x7fffff" in use_str:
                return pair, "mantissa_extraction"
            if "0x80000000" in use_str:
                return pair, "sign_extraction"
            if "0x7f800000" in use_str:
                return pair, "exponent_mask_check"
            return pair, "bitwise_on_float"
        # PHI merging float and int paths — typically from multi-path
        # float construction (branch merges different float representations)
        if c.use_opcode == "PHI":
            return pair, "phi_merge_type_conflict"
        # Integer comparison on float bits
        if c.use_opcode in ("ISETP",):
            return pair, "comparison_on_float_bits"
        return pair, "unclassified_f32_to_i32"

    # ── Int32 → Float32 conflicts ──────────────────────────────────
    if "Int32" in prev and "Float32" in new:
        pair = "Int32→Float32"
        # Constant materialization: memory load consumed as float
        if c.def_opcode in ("MOV", "SETZERO", "LDC", "LDG", "LDS"):
            return pair, "float_const_materialization"
        # LOP3 constructing float from parts (sign|exponent|mantissa)
        # Common pattern: LOP3 with 0x3f800000 (1.0f), 0x7f800000 (inf),
        # 0x80000000 (sign bit) → result fed to MUFU/FFMA
        if c.def_opcode == "LOP3":
            def_str = c.def_inst_str
            if "0x3f800000" in def_str:
                return pair, "float_reconstruction_1.0"
            if "0x7f800000" in def_str:
                return pair, "inf_construction"
            if "0x80000000" in def_str:
                return pair, "sign_injection"
            return pair, "float_reconstruction_bitwise"
        # IMAD constructing float via exponent manipulation
        # e.g., IMAD R31 = R74, -0x800000, R5 (subtract exponent bias)
        if c.def_opcode == "IMAD" and "0x800000" in c.def_inst_str:
            return pair, "exponent_bias_adjust"
        # IMAD.WIDE producing address bits reinterpreted as float (width issue)
        if c.def_opcode == "IMAD" and "WIDE" in c.def_inst_str:
            return pair, "width_reinterpret_from_wide"
        # End of fast division chain: IADD result → F2I
        if c.def_opcode in ("IADD3", "IADD32I", "IADD"):
            return pair, "fast_division_chain_end"
        # IMAD with other patterns
        if c.def_opcode == "IMAD":
            return pair, "int_arith_to_float_reinterpret"
        # PHI merge: different branches produce int vs float
        if c.def_opcode == "PHI":
            return pair, "phi_merge_type_conflict"
        # LEA used for bit-field construction (exponent << shift + mantissa)
        if c.def_opcode == "LEA":
            return pair, "bitfield_float_construction"
        return pair, "unclassified_i32_to_f32"

    # ── Other type pairs ───────────────────────────────────────────
    if "Float64" in prev and "Int64" in new:
        return "Float64→Int64", "double_int_reinterpret"

    if "Int64" in prev and "Float64" in new:
        return "Int64→Float64", "int_to_double_reinterpret"

    if "Int32" in prev and "Bool" in new:
        return "Int32→Bool", "int_to_predicate"

    if "Bool" in prev and "Int32" in new:
        return "Bool→Int32", "predicate_to_int"

    if "Float32" in prev and "Bool" in new:
        return "Float32→Bool", "float_to_predicate"

    return f"{prev}→{new}", "unclassified"


# ─── Auto-discovery ─────────────────────────────────────────────────────

TEST_DIR = PROJECT_ROOT / "test"

# Domain classification rules for addrtest prefixes.
# Checked in order; first match wins.
_ADDRTEST_DOMAIN_RULES = [
    # Prefixes that map to "Address/Memory"
    (re.compile(r"^(A\d|H|GD|D2|DS|BF)"), "Address/Memory"),
    # AC → Async Copy  (before general A/C rules)
    (re.compile(r"^AC"), "Async Copy"),
    # B2x → Bit Manipulation
    (re.compile(r"^B2"), "Bit Manipulation"),
    # CB → Constant Bank  (before general C rule)
    (re.compile(r"^CB"), "Constant Bank"),
    # C → Control Flow  (after CB)
    (re.compile(r"^C"), "Control Flow"),
    # DM or MU → Math/Division
    (re.compile(r"^(DM|MU)"), "Math/Division"),
    # F3 → FP16/Conversion
    (re.compile(r"^F3"), "FP16/Conversion"),
    # IM → Int/Float Mixed
    (re.compile(r"^IM"), "Int/Float Mixed"),
]


def _classify_domain(suite: str, test_name: str) -> str:
    """Return a domain string for a kernel based on its suite and name."""
    if suite == "cuobjdump":
        return "CUDA SDK"
    # addrtest: walk the prefix rules
    for pattern, domain in _ADDRTEST_DOMAIN_RULES:
        if pattern.match(test_name):
            return domain
    return "Other"


def discover_kernels():
    """Scan test/cuobjdump/ and test/addrtest/ for SM90 sass+elf pairs.

    Returns a list of (suite_test, test_name, domain) tuples and prints
    discovery statistics.
    """
    kernels = []
    suite_counts = Counter()

    for suite in ("cuobjdump", "addrtest"):
        suite_dir = TEST_DIR / suite
        if not suite_dir.is_dir():
            print(f"WARNING: {suite_dir} not found, skipping")
            continue

        for test_dir in sorted(suite_dir.iterdir()):
            if not test_dir.is_dir():
                continue
            test_name = test_dir.name
            sass_file = test_dir / f"{test_name}_sm90.sass"
            elf_file = test_dir / f"{test_name}_sm90.elf"
            if sass_file.exists() and elf_file.exists():
                suite_test = f"{suite}/{test_name}"
                domain = _classify_domain(suite, test_name)
                kernels.append((suite_test, test_name, domain))
                suite_counts[suite] += 1

    # Print discovery summary
    total = sum(suite_counts.values())
    print(f"Auto-discovered {total} SM90 kernels:")
    for suite, count in sorted(suite_counts.items()):
        print(f"  {suite}: {count}")
    print()

    return kernels


def main():
    patch_typeanalysis_for_logging()

    kernels = discover_kernels()

    all_conflicts = []
    pair_counter = Counter()
    pattern_counter = Counter()
    domain_pattern_counter = defaultdict(Counter)  # domain -> pattern -> count

    for suite_test, test_name, domain in kernels:
        test_dir = TEST_DIR / suite_test

        sass_file = test_dir / f"{test_name}_sm90.sass"
        elf_file = test_dir / f"{test_name}_sm90.elf"

        print(f"\n{'='*70}")
        print(f"KERNEL: {suite_test}/{test_name}  [{domain}]")
        print(f"{'='*70}")

        try:
            conflicts, module = lift_and_collect_conflicts(sass_file, elf_file)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        print(f"\n  Total conflicts: {len(conflicts)}")

        if not conflicts:
            print("  (no type conflicts)")
            continue

        for c in conflicts:
            pair, pattern = classify_conflict(c)
            pair_counter[pair] += 1
            pattern_counter[pattern] += 1
            domain_pattern_counter[domain][pattern] += 1
            all_conflicts.append({
                "kernel": f"{suite_test}/{test_name}",
                "domain": domain,
                "reg": c.reg or "?",
                "pair": pair,
                "pattern": pattern,
                "def_opcode": c.def_opcode,
                "use_opcode": c.use_opcode,
                "prev_type": c.prev_type,
                "new_type": c.new_type,
                "def_inst": c.def_inst_str[:80],
                "use_inst": c.use_inst_str[:80],
            })

        if len(conflicts) <= 20:
            print(f"\n  Conflicts in {test_name}:")
            for c in conflicts:
                pair, pattern = classify_conflict(c)
                reg = c.reg or "?"
                print(f"    {reg:12s} {pair:20s} [{pattern}]")

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"Total conflicts across all kernels: {len(all_conflicts)}")

    print(f"\nBy type pair:")
    for pair, count in pair_counter.most_common():
        print(f"  {pair:25s}: {count}")

    print(f"\nBy pattern:")
    for pattern, count in pattern_counter.most_common():
        print(f"  {pattern:35s}: {count}")

    print(f"\nBy domain:")
    for domain in sorted(domain_pattern_counter):
        domain_total = sum(domain_pattern_counter[domain].values())
        print(f"\n  {domain} ({domain_total} conflicts):")
        for pattern, count in domain_pattern_counter[domain].most_common():
            print(f"    {pattern:35s}: {count}")

    # Build by_domain dict for JSON output
    by_domain = {}
    for domain in sorted(domain_pattern_counter):
        by_domain[domain] = {
            "total": sum(domain_pattern_counter[domain].values()),
            "patterns": dict(domain_pattern_counter[domain]),
        }

    # Write JSON for further analysis
    out_path = Path(os.environ.get("SLIFTER_FIG_DATA", str(PROJECT_ROOT / "reproduce/figures/_data"))) / "conflict_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "total_conflicts": len(all_conflicts),
                "by_type_pair": dict(pair_counter),
                "by_pattern": dict(pattern_counter),
                "by_domain": by_domain,
            },
            "conflicts": all_conflicts,
        }, f, indent=2)
    print(f"\nDetailed results written to: {out_path}")


if __name__ == "__main__":
    main()
