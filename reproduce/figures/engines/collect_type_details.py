#!/usr/bin/env python3
"""Collect detailed type composition and unresolved-value flow stats.

New metrics beyond collect_eval_stats.py:
  T1: Resolved type composition — count of values resolved to each concrete type
  T2: Unresolved value downstream — do unresolved values flow into FP or INT consumers

Usage:
    python -m reproduce.figures.engines.collect_type_details --suite hecbench --output reproduce/figures/_data/type_details/ -j 8
    python -m reproduce.figures.engines.collect_type_details --suite all --output reproduce/figures/_data/type_details/ -j 8
"""

import argparse
import csv
import json
import multiprocessing
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.figures.engines.collect_eval_stats import discover_kernels, _KERNEL_TIMEOUT
from reproduce.figures.engines.bench_domains import get_domain
from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf
from transform.typeanalysis import TypeAnalysis, TOP, BOTTOM
from transform.typeanalysis import BOOL, INT32, FLOAT32, INT64, FLOAT64, FLOAT16, INT128

# Canonical type names for single-element frozensets
TYPE_NAMES = {
    frozenset({"Bool"}): "Bool",
    frozenset({"Int32"}): "Int32",
    frozenset({"Float32"}): "Float32",
    frozenset({"Int64"}): "Int64",
    frozenset({"Float64"}): "Float64",
    frozenset({"Float16"}): "Float16",
    frozenset({"Int128"}): "Int128",
}

# FP-consuming opcodes (instructions that interpret operands as floats)
FP_OPCODES = {
    "FADD", "FMUL", "FFMA", "FMNMX", "FSET", "FSETP", "FCMP",
    "MUFU", "DADD", "DMUL", "DFMA", "DMNMX", "DSET", "DSETP",
    "HADD2", "HMUL2", "HFMA2", "HSET2", "HSETP2", "HMNMX2",
    "F2I", "F2F", "I2F", "I2FP", "F2FP",
    "STG.E",  # if storing float value
}


def collect_type_details_for_kernel(name, sass_path, elf_path):
    """Collect detailed type stats for one kernel."""
    result = {"name": name, "status": "ok", "error": None}

    try:
        elf = Elf(str(elf_path))
        sass_text = Path(sass_path).read_text()
        sass_parser = SaSSParser(sass_text, elf=elf)
        module = Module(name, sass_parser)

        # Run transforms up to TypeAnalysis (same pipeline as collect_eval_stats)
        from transform.build_cfg import BuildCFG
        from transform.resolve_brx import ResolveBRX
        from transform.handle_call import HandleCall
        from transform.cfg_simplify import CFGSimplify
        from transform.cg_patterns import CGPatterns
        from transform.warp_patterns import WarpPatterns
        from transform.sr_substitute import SRSubstitute
        from transform.opmod_transform import OpModTransform
        from transform.predicated_to_sel import PredicatedToSel
        from transform.ssa import SSA
        from transform.reg_remap import RegRemap
        from transform.defuse_analysis import DefUseAnalysis
        from transform.xmad_to_imad import XmadToImad
        from transform.fp_hack import FPHack
        from transform.set_zero import SetZero
        from transform.dce import DCE
        from transform.mov_eliminate import MovEliminate
        from types import SimpleNamespace

        def _make_passes():
            return [
                BuildCFG(), ResolveBRX(), HandleCall(), CFGSimplify(),
                CGPatterns(), WarpPatterns(), SRSubstitute(), OpModTransform(),
                PredicatedToSel(), SSA(), RegRemap(), DefUseAnalysis(),
                XmadToImad(), FPHack(), RegRemap(), SetZero(),
                DefUseAnalysis(), MovEliminate(), DefUseAnalysis(), DCE(),
                CFGSimplify(), DefUseAnalysis(), RegRemap(), DefUseAnalysis(),
                DCE(), CFGSimplify(), DefUseAnalysis(),
            ]

        all_funcs = list(module.functions)
        if len(all_funcs) > 1:
            transformed = []
            for func in all_funcs:
                single = SimpleNamespace(
                    name=module.name, isa=module.isa,
                    elf=module.elf, functions=[func])
                for t in _make_passes():
                    t.apply(single)
                transformed.append(single.functions[0])
            module.functions = transformed
        else:
            for t in _make_passes():
                t.apply(module)

        # Run TypeAnalysis and collect resolved types
        ta = TypeAnalysis()
        # _INIT_TYPE_STATS_PATCHED — type_stats is set in process_func() not
        # __init__(); we bypass process_func() so we initialize the dict here.
        ta.type_stats = {"unique_resolved": 0, "ambiguous_int32": 0,
                         "ambiguous_int64": 0, "ambiguous_other": 0,
                         "empty_default": 0, "reg_group": 0, "total": 0}
        ta.type_detail_counts = {}
        type_counts = Counter()  # T1: resolved type composition
        unresolved_consumers = Counter()  # T2: what consumes unresolved values

        for func in module.functions:
            ta.conflicts = set()
            work_list = ta.traverse_cfg(func)
            optypes = {}

            # Phase 1: Static seeding
            for bb in work_list:
                for inst in bb.instructions:
                    ta.resolve_types(inst, optypes)

            # Phase 2: Iterative propagation
            changed = True
            iteration = 0
            while changed:
                changed = False
                for bb in work_list:
                    old = optypes.copy()
                    for inst in bb.instructions:
                        ta.propagate_types(inst, optypes)
                    if old != optypes:
                        changed = True
                iteration += 1
                if iteration > 5:
                    break

            # Match TypeAnalysis.process_func() and assign final type_descs to
            # both child regs and synthetic reg-groups before collecting T1.
            for bb in work_list:
                for inst in bb.instructions:
                    for op in inst.operands:
                        if op.is_reg_group:
                            for child_op in op.reg_group:
                                child_op.type_desc = ta.get_type_desc(child_op, optypes)
                        op.type_desc = ta.get_type_desc(op, optypes)

            # T1: Count resolved types from the scalar optype map first.
            for key, val in optypes.items():
                if val == TOP or len(val) >= len(TOP):
                    type_counts["unresolved"] += 1
                elif len(val) == 1:
                    type_name = TYPE_NAMES.get(val, str(val))
                    type_counts[type_name] += 1
                elif val == INT32 | FLOAT32:
                    type_counts["Num32"] += 1
                elif val == INT64 | FLOAT64:
                    type_counts["Num64"] += 1
                else:
                    type_counts["other_set"] += 1

            # Wide reg-groups (notably Int128 quads) are flattened back to their
            # child scalar regs in optypes, which hides them from T1. Recover the
            # logical group value here by replacing the counted child entries with
            # one group-level entry keyed by the final type_desc.
            wide_groups = {}
            for bb in work_list:
                for inst in bb.instructions:
                    for op in inst.operands:
                        if not op.is_reg_group:
                            continue
                        group_key = op.get_ir_name(None)
                        if not group_key:
                            continue
                        wide_groups[group_key] = {
                            "group_type": op.get_type_desc(),
                            "child_types": [child.get_type_desc() for child in op.reg_group],
                        }

            for payload in wide_groups.values():
                group_type = payload["group_type"]
                if group_type in ("Int64", "Float64"):
                    continue
                for child_type in payload["child_types"]:
                    if type_counts.get(child_type, 0) > 0:
                        type_counts[child_type] -= 1
                type_counts[group_type] += 1

            # T2: For unresolved values, check what instructions consume them
            unresolved_ops = set()
            for key, val in optypes.items():
                if val == TOP or len(val) >= len(TOP):
                    unresolved_ops.add(key)

            for bb in work_list:
                for inst in bb.instructions:
                    opcode = inst.opcodes[0]
                    uses = inst.get_uses()
                    for use_op in uses:
                        if use_op in unresolved_ops:
                            if opcode in FP_OPCODES or opcode.startswith("F") or opcode.startswith("H") or opcode.startswith("D"):
                                unresolved_consumers["fp_consumer"] += 1
                            else:
                                unresolved_consumers["int_consumer"] += 1

        result["type_composition"] = dict(type_counts)
        result["unresolved_consumers"] = dict(unresolved_consumers)

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect detailed type composition stats")
    parser.add_argument("--suite", required=True, help="Benchmark suite")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("-j", "--parallel", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    suites = ["hecbench", "cuobjdump", "cublas", "cutlass", "flash_attn",
              "cudnn_cnn", "cudnn_ops", "cudnn_adv"] if args.suite == "all" else [args.suite]

    for suite in suites:
        kernels = discover_kernels(suite, output_dir=output_dir)
        if args.limit > 0:
            kernels = kernels[:args.limit]
        if not kernels:
            print(f"No kernels for {suite}")
            continue

        print(f"\n=== {suite}: {len(kernels)} kernels ===")
        all_results = []

        if args.parallel > 1:
            with multiprocessing.Pool(args.parallel, maxtasksperchild=1) as pool:
                async_results = []
                for n, s, e in kernels:
                    ar = pool.apply_async(collect_type_details_for_kernel, (n, s, e))
                    async_results.append((n, ar))
                for i, (kname, ar) in enumerate(async_results):
                    try:
                        r = ar.get(timeout=_KERNEL_TIMEOUT)
                    except Exception as exc:
                        r = {"name": kname, "status": "error", "error": str(exc)[:100]}
                    all_results.append(r)
                    if (i + 1) % 50 == 0 or i + 1 == len(kernels):
                        print(f"  [{i+1}/{len(kernels)}]")
        else:
            for i, (n, s, e) in enumerate(kernels):
                r = collect_type_details_for_kernel(n, s, e)
                all_results.append(r)
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(kernels)}]")

        # Write type composition CSV
        type_csv = output_dir / f"{suite}_type_composition.csv"
        all_types = set()
        for r in all_results:
            if "type_composition" in r:
                all_types.update(r["type_composition"].keys())
        type_cols = sorted(all_types)

        with open(type_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "domain", "suite"] + type_cols)
            for r in all_results:
                if "type_composition" not in r:
                    continue
                domain = get_domain(r["name"])
                row = [r["name"], domain, suite]
                for t in type_cols:
                    row.append(r["type_composition"].get(t, 0))
                w.writerow(row)

        # Write unresolved consumer CSV
        unres_csv = output_dir / f"{suite}_unresolved_flow.csv"
        with open(unres_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kernel", "domain", "suite", "fp_consumer", "int_consumer"])
            for r in all_results:
                if "unresolved_consumers" not in r:
                    continue
                domain = get_domain(r["name"])
                uc = r["unresolved_consumers"]
                w.writerow([r["name"], domain, suite, uc.get("fp_consumer", 0), uc.get("int_consumer", 0)])

        ok = sum(1 for r in all_results if r["status"] == "ok")
        print(f"  {suite}: {ok}/{len(all_results)} OK")
        print(f"  -> {type_csv}")
        print(f"  -> {unres_csv}")


if __name__ == "__main__":
    main()
