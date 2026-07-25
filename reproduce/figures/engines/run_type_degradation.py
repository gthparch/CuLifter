#!/usr/bin/env python3
"""Type seeding degradation study.

Progressively remove type table entries and measure:
  1. Lift success rate (can LLVM IR be generated?)
  2. Type resolution rate (% values with concrete types)

Usage:
    python -m reproduce.figures.engines.run_type_degradation -j 8 --output reproduce/figures/_data/degradation/
"""

import argparse
import csv
import json
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.figures.engines.collect_eval_stats import discover_kernels
from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf
from transform.typeanalysis import TypeAnalysis, TOP

REMOVAL_RATES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # percent removed

# Global: computed once, reused across all runs
_REMOVAL_ORDER = None


def compute_removal_order(kernels):
    """Rank seeding opcodes by importance (kernel coverage). Least important first."""
    from collections import Counter
    ta = TypeAnalysis()
    opcode_kernels = Counter()
    for name, sass_path, elf_path in kernels:
        try:
            elf = Elf(str(elf_path))
            text = Path(sass_path).read_text()
            parser = SaSSParser(text, elf=elf)
            module = Module(name, parser)
            seen = set()
            for func in module.functions:
                for inst in func.instructions:
                    entry = ta._find_type_table(inst.opcodes)
                    if entry is not None:
                        all_top = all(t == TOP for t in entry[0]) and all(t == TOP for t in entry[1])
                        if not all_top:
                            base = inst.opcodes[0]
                            best_key, best_len = None, -1
                            for key in ta.instruction_type_table:
                                parts = key.split('.')
                                if parts[0] == base:
                                    key_mods = parts[1:]
                                    if ta._is_subseq(key_mods, inst.opcodes[1:]) and len(key_mods) > best_len:
                                        best_key, best_len = key, len(key_mods)
                            if best_key:
                                seen.add(best_key)
            for k in seen:
                opcode_kernels[k] += 1
        except:
            pass
    # Unseen entries first (zero importance), then by ascending kernel count
    all_seeding = get_type_table_keys(ta)
    unseen = [k for k in all_seeding if k not in opcode_kernels]
    ranked = unseen + sorted(opcode_kernels.keys(), key=lambda k: opcode_kernels[k])
    return ranked


def get_type_table_keys(ta):
    """Extract all seeding keys from TypeAnalysis instruction_type_table."""
    keys = []
    for key in ta.instruction_type_table:
        defs, uses = ta.instruction_type_table[key]
        all_top = all(t == TOP for t in defs) and all(t == TOP for t in uses)
        if not all_top:
            keys.append(key)
    return keys


def make_degraded_ta(removal_pct, removal_order=None):
    """Create a TypeAnalysis with removal_pct% of seeding entries blanked to TOP.

    Uses ordered removal (least important first) if removal_order is provided.
    """
    ta = TypeAnalysis()
    if removal_pct == 0:
        return ta

    all_keys = get_type_table_keys(ta)
    n_remove = int(len(all_keys) * removal_pct / 100)

    if removal_order is not None:
        to_remove = set(removal_order[:n_remove])
    else:
        import random
        to_remove = set(random.Random(42).sample(all_keys, n_remove))

    for key in to_remove:
        if key in ta.instruction_type_table:
            defs, uses = ta.instruction_type_table[key]
            ta.instruction_type_table[key] = ([TOP] * len(defs), [TOP] * len(uses))

    return ta


def run_one_kernel(name, sass_path, elf_path, removal_pct, removal_order=None):
    """Lift one kernel with degraded type table. Returns result dict."""
    result = {
        "name": name, "removal_pct": removal_pct,
        "lift_ok": False, "compile_ok": False,
        "total_values": 0, "resolved_values": 0, "resolution_pct": 0.0,
    }

    try:
        elf = Elf(str(elf_path))
        sass_text = Path(sass_path).read_text()
        sass_parser = SaSSParser(sass_text, elf=elf)
        module = Module(name, sass_parser)

        # Run transform pipeline
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

        passes = [
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
                for t in passes:
                    t.apply(single)
                transformed.append(single.functions[0])
            module.functions = transformed
        else:
            for t in passes:
                t.apply(module)

        # Run degraded TypeAnalysis
        ta = make_degraded_ta(removal_pct, removal_order)
        total_values = 0
        resolved_values = 0

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

            # Count resolution
            for key, val in optypes.items():
                total_values += 1
                if val != TOP and len(val) < len(TOP):
                    resolved_values += 1

        result["total_values"] = total_values
        result["resolved_values"] = resolved_values
        result["resolution_pct"] = round(100 * resolved_values / max(total_values, 1), 2)

        # Try lifting in-process with degraded type table
        try:
            # Apply degraded types to operands
            ta2 = make_degraded_ta(removal_pct, removal_order)
            ta2.apply(module)

            # Attempt LLVM IR generation
            import io, contextlib
            from lift.lifter_cpu import CpuLifter as X86Lifter
            lifter = X86Lifter(elf=elf, verbose=False)
            lifter.llvm_module = lifter.ir.Module(name)
            lifter.intrinsics = {}
            lifter.ir_functions = {}
            lifter.setup_module(lifter.llvm_module)
            lifter._declare_ir_functions(module.functions, lifter.llvm_module)
            with contextlib.redirect_stdout(io.StringIO()):
                for func in module.functions:
                    lifter.curr_function = func
                    lifter._lift_function(func, lifter.llvm_module)
            result["lift_ok"] = True
        except Exception as e:
            result["lift_ok"] = False
            result["lift_error"] = f"{type(e).__name__}: {str(e)[:100]}"

    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def main():
    parser = argparse.ArgumentParser(description="Type seeding degradation study")
    parser.add_argument("-j", "--parallel", type=int, default=1)
    parser.add_argument("--output", default="reproduce/figures/_data/degradation")
    parser.add_argument("--suite", default="hecbench")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rates", default=",".join(str(r) for r in REMOVAL_RATES),
                        help="Comma-separated removal percentages")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rates = [int(r) for r in args.rates.split(",")]
    kernels = discover_kernels(args.suite, output_dir=output_dir)
    if args.limit > 0:
        kernels = kernels[:args.limit]

    print(f"Type degradation: {len(kernels)} kernels × {len(rates)} rates = {len(kernels) * len(rates)} runs")
    print(f"Rates: {rates}")

    # Check how many seeding entries exist
    ta = TypeAnalysis()
    all_keys = get_type_table_keys(ta)
    print(f"Type table: {len(ta.instruction_type_table)} total entries, {len(all_keys)} seeding entries")

    # Compute ordered removal ranking
    print("Computing opcode importance ranking...")
    removal_order = compute_removal_order(kernels)
    print(f"Ranked {len(removal_order)} seeding entries (least→most important)")
    print()

    all_results = []
    tasks = [(name, sass, elf, rate) for name, sass, elf in kernels for rate in rates]

    if args.parallel > 1:
        import multiprocessing
        with multiprocessing.Pool(args.parallel, maxtasksperchild=1) as pool:
            async_results = []
            for name, sass, elf, rate in tasks:
                ar = pool.apply_async(run_one_kernel, (name, sass, elf, rate, removal_order))
                async_results.append(((name, rate), ar))
            for i, ((name, rate), ar) in enumerate(async_results):
                try:
                    # Some kernels (e.g. grrt-cuda ~145s) legitimately lift slowly;
                    # 300s under heavy -j contention spuriously timed them out and
                    # recorded false lift failures. Allow override; default 900s.
                    r = ar.get(timeout=int(os.environ.get("DEGRADE_TIMEOUT", "900")))
                except Exception as e:
                    r = {"name": name, "removal_pct": rate, "lift_ok": False,
                         "total_values": 0, "resolved_values": 0, "resolution_pct": 0,
                         "error": str(e)[:100]}
                all_results.append(r)
                if (i + 1) % 100 == 0 or i + 1 == len(tasks):
                    print(f"  [{i+1}/{len(tasks)}]")
    else:
        for i, (name, sass, elf, rate) in enumerate(tasks):
            r = run_one_kernel(name, sass, elf, rate, removal_order)
            all_results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(tasks)}]")

    # Write raw CSV
    csv_path = output_dir / "degradation_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "removal_pct", "lift_ok",
                                           "total_values", "resolved_values", "resolution_pct"])
        w.writeheader()
        for r in all_results:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # Summary per rate
    print(f"\n{'='*60}")
    print(f"Type Seeding Degradation Results")
    print(f"{'='*60}")
    summary = {}
    for rate in rates:
        rr = [r for r in all_results if r.get("removal_pct") == rate]
        n_lift = sum(1 for r in rr if r.get("lift_ok"))
        total = len(rr)
        avg_res = sum(r.get("resolution_pct", 0) for r in rr) / max(total, 1)
        summary[rate] = {
            "removal_pct": rate,
            "total": total,
            "lift_ok": n_lift,
            "lift_rate": round(100 * n_lift / max(total, 1), 1),
            "avg_resolution": round(avg_res, 1),
        }
        print(f"  {rate:3d}% removed: {n_lift:3d}/{total} lift OK ({summary[rate]['lift_rate']:.1f}%)  "
              f"resolution: {avg_res:.1f}%")

    with open(output_dir / "degradation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {csv_path}")
    print(f"Summary: {output_dir / 'degradation_summary.json'}")


if __name__ == "__main__":
    main()
