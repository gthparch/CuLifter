#!/usr/bin/env python3
"""Collect evaluation statistics for paper figures.

Runs the SLifter pipeline on each kernel and collects:
  D1: instruction type classification (seeding vs transparent vs unknown)
  D2: values resolved by seeding vs propagation vs unresolved
  D3: propagation reach (hop count from seed to resolved value)
  D4: type conflicts and bitcast count
  D5: convergence iteration count
  D6: control flow pattern frequency

Usage:
    # Collect stats for hecbench (uses pre-built SASS from cluster artifacts)
    python -m reproduce.figures.engines.collect_eval_stats --suite hecbench --output reproduce/figures/_data/

    # Collect for synthetic tests
    python -m reproduce.figures.engines.collect_eval_stats --suite addrtest --output reproduce/figures/_data/

    # Collect for specific kernels only
    python -m reproduce.figures.engines.collect_eval_stats --suite hecbench --kernels relu,softmax --output reproduce/figures/_data/

    # Lift-only mode (no execution, just stats)
    python -m reproduce.figures.engines.collect_eval_stats --suite hecbench --output reproduce/figures/_data/ --lift-only
"""

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf
from transform.typeanalysis import TypeAnalysis, TOP, BOTTOM, NUM32, BOOL, INT32, FLOAT32
from reproduce.figures.engines.bench_domains import get_domain

# ---------------------------------------------------------------------------
# Suite discovery — reads the download-only benchmark/ bundle
# (reproduce/figures/engines/bench_bundle.py is the single source of truth).
# ---------------------------------------------------------------------------

import os as _os
TEST_DIR = PROJECT_ROOT / "test"
from reproduce.figures.engines.bench_bundle import discover_kernels, FIG_SUITES


# ---------------------------------------------------------------------------
# D1: Instruction type classification
# ---------------------------------------------------------------------------

def classify_instructions(module, ta):
    """Classify instructions as seeding, transparent, or unclassified."""
    stats = Counter()
    for func in module.functions:
        for inst in func.instructions:
            opcode = inst.opcodes[0]
            # Skip pseudo-instructions
            if opcode in ("PHI", "PHI64", "SWITCH", "SEL"):
                stats["pseudo"] += 1
                continue
            # Check type table (seeding)
            table_entry = ta._find_type_table(inst.opcodes)
            if table_entry is not None:
                # Check if any type is not TOP
                all_top = all(t == TOP for t in table_entry[0]) and all(t == TOP for t in table_entry[1])
                if not all_top:
                    stats["seeding"] += 1
                else:
                    stats["transparent_in_table"] += 1
            elif opcode in ta.propagate_table:
                stats["transparent"] += 1
            else:
                stats["unclassified"] += 1
            stats["total"] += 1
    return dict(stats)


# ---------------------------------------------------------------------------
# D2 + D3: Type propagation stats (seeded vs propagated vs unresolved + reach)
# ---------------------------------------------------------------------------

def collect_type_propagation_stats(module, ta):
    """Run type analysis with instrumentation to collect propagation stats.

    Returns dict with:
      - seeded: # values typed during static phase
      - propagated: # values typed during iterative phase
      - unresolved: # values still ambiguous after fixpoint
      - iterations: # fixpoint iterations
      - conflicts: # type conflicts detected
      - reach_histogram: {hop_count: count}
    """
    stats = {
        "seeded": 0,
        "propagated": 0,
        "unresolved": 0,
        "iterations": 0,
        "conflicts": 0,
        "reach_histogram": defaultdict(int),
    }

    for func in module.functions:
        ta.conflicts = set()
        work_list = ta.traverse_cfg(func)
        optypes = {}

        # Phase 1: Static seeding
        for bb in work_list:
            for inst in bb.instructions:
                ta.resolve_types(inst, optypes)

        # Snapshot after seeding
        seeded_regs = set()
        for key, val in optypes.items():
            reg = key.reg if hasattr(key, 'reg') else key
            if val != TOP and len(val) < len(TOP):
                seeded_regs.add(reg if isinstance(reg, str) else str(reg))

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

        stats["iterations"] = max(stats["iterations"], iteration)
        stats["conflicts"] += len(ta.conflicts)

        # Compare: what was resolved by seeding vs propagation
        for key, val in optypes.items():
            reg = key.reg if hasattr(key, 'reg') else key
            reg_str = reg if isinstance(reg, str) else str(reg)
            if val == TOP or len(val) >= len(TOP):
                stats["unresolved"] += 1
            elif reg_str in seeded_regs:
                stats["seeded"] += 1
            else:
                stats["propagated"] += 1

        # D3: Propagation reach via def-use BFS
        # For propagated values, walk backward to nearest seed
        for bb in work_list:
            for inst in bb.instructions:
                for op in inst.operands:
                    if not hasattr(op, 'reg') or not op.is_writable_reg:
                        continue
                    reg_str = op.reg
                    if reg_str in seeded_regs:
                        stats["reach_histogram"][0] += 1
                    elif op in optypes and optypes[op] != TOP:
                        # BFS backward through defining instructions
                        hops = _compute_reach(op, ta, work_list)
                        if hops >= 0:
                            stats["reach_histogram"][hops] += 1

    stats["reach_histogram"] = dict(stats["reach_histogram"])
    return stats


def _compute_reach(operand, ta, work_list):
    """BFS backward from operand through def-use to nearest typed seed. Returns hop count."""
    from collections import deque
    visited = set()
    queue = deque([(operand, 0)])

    while queue:
        op, hops = queue.popleft()
        if not hasattr(op, 'reg'):
            continue
        reg_id = id(op)
        if reg_id in visited:
            continue
        visited.add(reg_id)

        if hops > 20:  # safety bound
            return -1

        # Check if this operand's defining instruction is a seed
        if hasattr(op, 'defining_insts') and op.defining_insts:
            for def_inst in op.defining_insts:
                table_entry = ta._find_type_table(def_inst.opcodes)
                if table_entry is not None:
                    all_top = (all(t == TOP for t in table_entry[0]) and
                               all(t == TOP for t in table_entry[1]))
                    if not all_top:
                        return hops
                # Not a seed — continue backward
                for use_op in def_inst.get_uses():
                    if hasattr(use_op, 'reg') and use_op.is_writable_reg:
                        queue.append((use_op, hops + 1))

    return -1  # no seed found


# ---------------------------------------------------------------------------
# D4: Bitcast count from emitted LLVM IR
# ---------------------------------------------------------------------------

def count_bitcasts_in_ir(ll_path):
    """Count bitcast instructions in emitted LLVM IR file."""
    if not ll_path or not ll_path.exists():
        return 0
    text = ll_path.read_text()
    return text.count("bitcast ")


# ---------------------------------------------------------------------------
# D6: Control flow pattern stats
# ---------------------------------------------------------------------------

def collect_control_flow_stats(module):
    """Collect control flow pattern frequency from parsed module."""
    stats = Counter()
    for func in module.functions:
        stats["functions"] += 1
        stats["basic_blocks"] += len(func.blocks)
        for inst in func.instructions:
            stats["total_insts"] += 1
            if inst.pflag:
                stats["predicated"] += 1
            opcode = inst.opcodes[0]
            if opcode == "EXIT":
                if inst.pflag:
                    stats["predicated_exit"] += 1
                else:
                    stats["unconditional_exit"] += 1
            elif opcode == "BRA":
                if inst.pflag:
                    stats["predicated_bra"] += 1
                    # Check for dual-predicate branch
                    uses = inst.get_uses()
                    pred_uses = [u for u in uses if hasattr(u, 'is_predicate_reg') and u.is_predicate_reg]
                    if len(pred_uses) >= 1:  # P1 as convergence barrier
                        stats["dual_pred_bra"] += 1
                else:
                    stats["unconditional_bra"] += 1
            elif opcode == "BSSY":
                stats["bssy"] += 1
            elif opcode == "BSYNC":
                stats["bsync"] += 1
            elif opcode == "WARPSYNC":
                stats["warpsync"] += 1
            elif opcode == "SEL":
                stats["sel_inserted"] += 1
    return dict(stats)


# ---------------------------------------------------------------------------
# Main pipeline: parse → collect stats
# ---------------------------------------------------------------------------

import os as _os
_KERNEL_TIMEOUT = int(_os.environ.get("SLIFTER_EVAL_TIMEOUT", "120"))  # seconds per kernel


def _collect_stats_inner(name, sass_path, elf_path):
    """Inner function — runs in subprocess for timeout enforcement."""
    return collect_stats_for_kernel(name, sass_path, elf_path)


def collect_stats_for_kernel(name, sass_path, elf_path):
    """Parse a kernel and collect all stats. Returns dict or None on failure."""
    stats = {"name": name, "status": "ok", "error": None}

    try:
        # Parse (mirrors main.py). Two input shapes:
        #   binary cubin + SASS text (hecbench/flash/cublas/cudnn) -> Elf,
        #   nvdisasm TEXT dump only (cutlass/cuda_sdk, elf_path is None) -> Asm.
        sass_text = Path(sass_path).read_text()
        if elf_path is None:
            from utils.asm import Asm
            elf = Asm(sass_text, path=Path(sass_path))
        else:
            elf = Elf(str(elf_path))
        sass_parser = SaSSParser(sass_text, elf=elf)
        module = Module(name, sass_parser)

        # D6: Control flow stats (on raw parsed module)
        stats["control_flow"] = collect_control_flow_stats(module)

        # Run transforms up to (but not including) TypeAnalysis
        # Mirrors CpuLifter.get_transform_passes() pipeline
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

        # Pipeline matching CpuLifter up to (but not including) TypeAnalysis
        def _make_passes():
            return [
                BuildCFG(),
                ResolveBRX(),
                HandleCall(),
                CFGSimplify(),
                CGPatterns(),
                WarpPatterns(),
                SRSubstitute(),
                OpModTransform(),
                PredicatedToSel(),
                SSA(),
                RegRemap(),
                DefUseAnalysis(),
                XmadToImad(),
                FPHack(),
                RegRemap(),
                SetZero(),
                DefUseAnalysis(),
                MovEliminate(),
                DefUseAnalysis(),
                DCE(),
                CFGSimplify(),
                DefUseAnalysis(),
                RegRemap(),
                DefUseAnalysis(),
                DCE(),
                CFGSimplify(),
                DefUseAnalysis(),
            ]

        # Per-function transform pipeline (same optimization as lifter.py)
        from types import SimpleNamespace
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

        # D1: Instruction classification
        ta = TypeAnalysis()
        stats["type_classification"] = classify_instructions(module, ta)

        # D2 + D3 + D4(conflicts) + D5: Type propagation stats
        ta2 = TypeAnalysis()
        stats["type_propagation"] = collect_type_propagation_stats(module, ta2)

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        stats["traceback"] = traceback.format_exc()[-500:]

    return stats


# ---------------------------------------------------------------------------
# Aggregate and output
# ---------------------------------------------------------------------------

def aggregate_stats(all_stats, suite):
    """Aggregate per-kernel stats into summary tables."""
    summary = {
        "suite": suite,
        "total_kernels": len(all_stats),
        "successful": sum(1 for s in all_stats if s["status"] == "ok"),
        "failed": sum(1 for s in all_stats if s["status"] != "ok"),
    }

    # D1 aggregate
    d1_totals = Counter()
    for s in all_stats:
        if "type_classification" in s:
            for k, v in s["type_classification"].items():
                d1_totals[k] += v
    summary["type_classification_totals"] = dict(d1_totals)
    total = d1_totals.get("total", 1)
    summary["type_classification_pct"] = {
        "seeding_pct": round(100 * d1_totals.get("seeding", 0) / max(total, 1), 1),
        "transparent_pct": round(100 * (d1_totals.get("transparent", 0) +
                                        d1_totals.get("transparent_in_table", 0)) / max(total, 1), 1),
        "unclassified_pct": round(100 * d1_totals.get("unclassified", 0) / max(total, 1), 1),
    }

    # D2 aggregate
    d2_totals = Counter()
    for s in all_stats:
        if "type_propagation" in s:
            d2_totals["seeded"] += s["type_propagation"].get("seeded", 0)
            d2_totals["propagated"] += s["type_propagation"].get("propagated", 0)
            d2_totals["unresolved"] += s["type_propagation"].get("unresolved", 0)
            d2_totals["conflicts"] += s["type_propagation"].get("conflicts", 0)
    summary["type_propagation_totals"] = dict(d2_totals)

    # D3 aggregate (reach histogram)
    reach = Counter()
    for s in all_stats:
        if "type_propagation" in s and "reach_histogram" in s["type_propagation"]:
            for hops, count in s["type_propagation"]["reach_histogram"].items():
                reach[int(hops)] += count
    summary["reach_histogram"] = dict(sorted(reach.items()))

    # D5 aggregate (convergence)
    iterations = [s["type_propagation"]["iterations"]
                  for s in all_stats if "type_propagation" in s]
    if iterations:
        summary["convergence"] = {
            "max": max(iterations),
            "mean": round(sum(iterations) / len(iterations), 2),
            "distribution": dict(Counter(iterations)),
        }

    # D6 aggregate (control flow)
    cf_totals = Counter()
    for s in all_stats:
        if "control_flow" in s:
            for k, v in s["control_flow"].items():
                cf_totals[k] += v
    summary["control_flow_totals"] = dict(cf_totals)

    # Per-domain breakdown (hecbench only)
    if suite == "hecbench":
        by_domain = defaultdict(lambda: {"count": 0, "seeding": 0, "transparent": 0,
                                          "conflicts": 0, "predicated": 0, "predicated_exit": 0})
        for s in all_stats:
            domain = get_domain(s["name"])
            by_domain[domain]["count"] += 1
            if "type_classification" in s:
                by_domain[domain]["seeding"] += s["type_classification"].get("seeding", 0)
                by_domain[domain]["transparent"] += (
                    s["type_classification"].get("transparent", 0) +
                    s["type_classification"].get("transparent_in_table", 0))
            if "type_propagation" in s:
                by_domain[domain]["conflicts"] += s["type_propagation"].get("conflicts", 0)
            if "control_flow" in s:
                by_domain[domain]["predicated"] += s["control_flow"].get("predicated", 0)
                by_domain[domain]["predicated_exit"] += s["control_flow"].get("predicated_exit", 0)
        summary["by_domain"] = dict(by_domain)

    return summary


def write_csv(all_stats, output_dir, suite):
    """Write per-kernel CSV for easy import into plotting scripts."""
    import csv

    # Type classification CSV
    csv_path = output_dir / f"{suite}_type_classification.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "seeding", "transparent", "unclassified", "total"])
        for s in all_stats:
            if "type_classification" not in s:
                continue
            tc = s["type_classification"]
            domain = get_domain(s["name"])
            w.writerow([
                s["name"], domain,
                tc.get("seeding", 0),
                tc.get("transparent", 0) + tc.get("transparent_in_table", 0),
                tc.get("unclassified", 0),
                tc.get("total", 0),
            ])

    # Type propagation CSV
    csv_path = output_dir / f"{suite}_type_propagation.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "seeded", "propagated", "unresolved",
                     "conflicts", "iterations"])
        for s in all_stats:
            if "type_propagation" not in s:
                continue
            tp = s["type_propagation"]
            domain = get_domain(s["name"])
            w.writerow([
                s["name"], domain,
                tp.get("seeded", 0), tp.get("propagated", 0), tp.get("unresolved", 0),
                tp.get("conflicts", 0), tp.get("iterations", 0),
            ])

    # Control flow CSV
    csv_path = output_dir / f"{suite}_control_flow.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "total_insts", "predicated", "predicated_exit",
                     "dual_pred_bra", "bssy", "bsync", "sel_inserted", "basic_blocks"])
        for s in all_stats:
            if "control_flow" not in s:
                continue
            cf = s["control_flow"]
            domain = get_domain(s["name"])
            w.writerow([
                s["name"], domain,
                cf.get("total_insts", 0), cf.get("predicated", 0),
                cf.get("predicated_exit", 0), cf.get("dual_pred_bra", 0),
                cf.get("bssy", 0), cf.get("bsync", 0),
                cf.get("sel_inserted", 0), cf.get("basic_blocks", 0),
            ])

    # Propagation reach CSV (one row per kernel per hop-count)
    csv_path = output_dir / f"{suite}_propagation_reach.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "hops", "count"])
        for s in all_stats:
            if "type_propagation" not in s:
                continue
            domain = get_domain(s["name"])
            reach = s["type_propagation"].get("reach_histogram", {})
            for hops, count in sorted(reach.items(), key=lambda x: int(x[0])):
                w.writerow([s["name"], domain, hops, count])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_SUITES = ["hecbench", "cuobjdump", "ml", "cutlass", "flash_attn",
              "cublas", "cublaslt", "cudnn_cnn", "cudnn_ops", "cudnn_adv",
              "all_sdk"]


def run_suite(suite, output_dir, kernel_filter="", limit=0, parallel=1):
    """Collect stats for one suite. Returns (suite, all_stats)."""
    (output_dir / "per_kernel").mkdir(parents=True, exist_ok=True)

    kernels = discover_kernels(suite, kernel_filter, output_dir=output_dir)
    if not kernels:
        print(f"No kernels found for suite '{suite}'")
        return suite, []

    if limit > 0:
        kernels = kernels[:limit]

    print(f"\nCollecting stats for {len(kernels)} kernels from {suite}")
    print(f"Output: {output_dir}\n")

    all_stats = []

    if parallel > 1:
        import multiprocessing
        # Use imap_unordered for progress + per-kernel timeout via maxtasksperchild
        with multiprocessing.Pool(parallel, maxtasksperchild=1) as pool:
            args_list = [(n, s, e) for n, s, e in kernels]
            async_results = []
            for args in args_list:
                ar = pool.apply_async(collect_stats_for_kernel, args)
                async_results.append((args[0], ar))
            for i, (kname, ar) in enumerate(async_results):
                try:
                    stats = ar.get(timeout=_KERNEL_TIMEOUT)
                except multiprocessing.TimeoutError:
                    stats = {"name": kname, "status": "error",
                             "error": f"Timeout ({_KERNEL_TIMEOUT}s)"}
                except Exception as exc:
                    stats = {"name": kname, "status": "error",
                             "error": str(exc)[:200]}
                all_stats.append(stats)
                tag = "OK" if stats["status"] == "ok" else f"ERR: {stats.get('error', '')[:60]}"
                print(f"  [{i+1}/{len(kernels)}] {kname:<40s} {tag}")
                # Write per-kernel JSON incrementally
                safe_name = kname.replace("/", "_")
                with open(output_dir / "per_kernel" / f"{safe_name}.json", "w") as f:
                    json.dump(stats, f, indent=2, default=str)
    else:
        for i, (name, sass_path, elf_path) in enumerate(kernels):
            stats = collect_stats_for_kernel(name, sass_path, elf_path)
            all_stats.append(stats)

            tag = "OK" if stats["status"] == "ok" else f"ERR: {stats.get('error', '')[:60]}"
            print(f"  [{i+1}/{len(kernels)}] {name:<40s} {tag}")

            # Write per-kernel JSON (sanitize / in name for SDK libs)
            safe_name = name.replace("/", "_")
            with open(output_dir / "per_kernel" / f"{safe_name}.json", "w") as f:
                json.dump(stats, f, indent=2, default=str)

    # Aggregate
    summary = aggregate_stats(all_stats, suite)
    with open(output_dir / f"{suite}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Write CSVs
    write_csv(all_stats, output_dir, suite)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Suite: {suite} — {summary['successful']}/{summary['total_kernels']} OK")
    print(f"{'='*60}")

    return suite, all_stats


def print_summary(summary):
    """Print summary stats to stdout."""
    if "type_classification_pct" in summary:
        pct = summary["type_classification_pct"]
        print(f"\nD1: Instruction classification:")
        print(f"  Seeding:      {pct['seeding_pct']:5.1f}%")
        print(f"  Transparent:  {pct['transparent_pct']:5.1f}%")
        print(f"  Unclassified: {pct['unclassified_pct']:5.1f}%")

    if "type_propagation_totals" in summary:
        tp = summary["type_propagation_totals"]
        total_vals = tp.get("seeded", 0) + tp.get("propagated", 0) + tp.get("unresolved", 0)
        if total_vals > 0:
            print(f"\nD2: Type resolution:")
            print(f"  Seeded:     {tp['seeded']:6d} ({100*tp['seeded']/total_vals:.1f}%)")
            print(f"  Propagated: {tp['propagated']:6d} ({100*tp['propagated']/total_vals:.1f}%)")
            print(f"  Unresolved: {tp['unresolved']:6d} ({100*tp['unresolved']/total_vals:.1f}%)")
            print(f"  Conflicts:  {tp['conflicts']:6d}")

    if "convergence" in summary:
        cv = summary["convergence"]
        print(f"\nD5: Convergence: max={cv['max']}, mean={cv['mean']}")
        print(f"  Distribution: {cv['distribution']}")

    if "control_flow_totals" in summary:
        cf = summary["control_flow_totals"]
        print(f"\nD6: Control flow:")
        print(f"  Total insts:      {cf.get('total_insts', 0):8d}")
        print(f"  Predicated:       {cf.get('predicated', 0):8d}")
        print(f"  Predicated EXIT:  {cf.get('predicated_exit', 0):8d}")
        print(f"  Dual-pred BRA:    {cf.get('dual_pred_bra', 0):8d}")
        print(f"  BSSY/BSYNC:       {cf.get('bssy', 0):8d} / {cf.get('bsync', 0)}")
        print(f"  SEL inserted:     {cf.get('sel_inserted', 0):8d}")

    if "reach_histogram" in summary:
        rh = summary["reach_histogram"]
        print(f"\nD3: Propagation reach (hops → count):")
        for hops in sorted(rh.keys(), key=int):
            print(f"  {int(hops):2d} hops: {int(rh[hops]):6d}")


def main():
    parser = argparse.ArgumentParser(description="Collect evaluation statistics for paper")
    parser.add_argument("--suite", required=True,
                        choices=ALL_SUITES + ["all"],
                        help="Benchmark suite to process ('all' = all suites)")
    parser.add_argument("--output", required=True, help="Output directory for stats")
    parser.add_argument("--kernels", default="", help="Comma-separated kernel name filter")
    parser.add_argument("--limit", type=int, default=0, help="Max kernels per suite (0=all)")
    parser.add_argument("-j", "--parallel", type=int, default=1, help="Parallel workers")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.suite == "all":
        # Run all suites (excluding all_sdk which is a meta-suite)
        suites = ["hecbench", "cuobjdump", "ml", "cutlass", "flash_attn",
                   "cublas", "cublaslt", "cudnn_cnn", "cudnn_ops", "cudnn_adv"]
    else:
        suites = [args.suite]

    for suite in suites:
        suite_dir = output_dir / suite if len(suites) > 1 else output_dir
        suite_name, all_stats = run_suite(suite, suite_dir, args.kernels,
                                          args.limit, args.parallel)
        if all_stats:
            summary = json.loads((suite_dir / f"{suite}_summary.json").read_text())
            print_summary(summary)
            print(f"\nOutput: {suite_dir}/")

    if len(suites) > 1:
        print(f"\n{'='*60}")
        print(f"All suites complete. Results in {output_dir}/")
        print(f"{'='*60}")

    if "type_classification_pct" in summary:
        pct = summary["type_classification_pct"]
        print(f"\nD1: Instruction classification:")
        print(f"  Seeding:      {pct['seeding_pct']:5.1f}%")
        print(f"  Transparent:  {pct['transparent_pct']:5.1f}%")
        print(f"  Unclassified: {pct['unclassified_pct']:5.1f}%")

    if "type_propagation_totals" in summary:
        tp = summary["type_propagation_totals"]
        total_vals = tp.get("seeded", 0) + tp.get("propagated", 0) + tp.get("unresolved", 0)
        if total_vals > 0:
            print(f"\nD2: Type resolution:")
            print(f"  Seeded:     {tp['seeded']:6d} ({100*tp['seeded']/total_vals:.1f}%)")
            print(f"  Propagated: {tp['propagated']:6d} ({100*tp['propagated']/total_vals:.1f}%)")
            print(f"  Unresolved: {tp['unresolved']:6d} ({100*tp['unresolved']/total_vals:.1f}%)")
            print(f"  Conflicts:  {tp['conflicts']:6d}")

    if "convergence" in summary:
        cv = summary["convergence"]
        print(f"\nD5: Convergence: max={cv['max']}, mean={cv['mean']}")
        print(f"  Distribution: {cv['distribution']}")

    if "control_flow_totals" in summary:
        cf = summary["control_flow_totals"]
        print(f"\nD6: Control flow:")
        print(f"  Total insts:      {cf.get('total_insts', 0):8d}")
        print(f"  Predicated:       {cf.get('predicated', 0):8d}")
        print(f"  Predicated EXIT:  {cf.get('predicated_exit', 0):8d}")
        print(f"  Dual-pred BRA:    {cf.get('dual_pred_bra', 0):8d}")
        print(f"  BSSY/BSYNC:       {cf.get('bssy', 0):8d} / {cf.get('bsync', 0)}")
        print(f"  SEL inserted:     {cf.get('sel_inserted', 0):8d}")

    if "reach_histogram" in summary:
        rh = summary["reach_histogram"]
        print(f"\nD3: Propagation reach (hops → count):")
        for hops in sorted(rh.keys(), key=int):
            print(f"  {int(hops):2d} hops: {int(rh[hops]):6d}")

    print(f"\nOutput files:")
    print(f"  {output_dir / f'{args.suite}_summary.json'}")
    print(f"  {output_dir / f'{args.suite}_type_classification.csv'}")
    print(f"  {output_dir / f'{args.suite}_type_propagation.csv'}")
    print(f"  {output_dir / f'{args.suite}_control_flow.csv'}")
    print(f"  {output_dir / f'{args.suite}_propagation_reach.csv'}")


if __name__ == "__main__":
    main()
