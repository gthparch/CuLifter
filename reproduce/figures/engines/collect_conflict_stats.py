#!/usr/bin/env python3
"""Collect type-conflict pattern analysis.

This script is intentionally narrower than collect_eval_stats.py: it runs the
same transform pipeline, but only keeps the data needed for conflict taxonomy
analysis and the "conflict pattern breakdown by domain" figure/table.

For each kernel we record:
  - whether the kernel finished analysis
  - the number of type conflicts
  - per-conflict classification records:
      (domain, pair, pattern, def_opcode, use_opcode, reg)

Per suite we aggregate:
  - total conflicts
  - conflicts by type-pair
  - conflicts by pattern
  - conflicts by domain, with per-pattern breakdown
  - flat conflict detail rows for later merging/plotting

Usage:
    python -m reproduce.figures.engines.collect_conflict_stats --suite hecbench --output reproduce/figures/_data/
    python -m reproduce.figures.engines.collect_conflict_stats --suite all -j 8 --output reproduce/figures/_data/
"""

import argparse
import contextlib
import json
import re
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf
from transform.typeanalysis import TypeAnalysis
from reproduce.figures.engines.bench_domains import get_domain
from reproduce.figures.engines.analyze_conflicts import (
    classify_conflict, patch_typeanalysis_for_logging,
)

# Monkey-patch TypeAnalysis at import time so every worker gets it
patch_typeanalysis_for_logging()

# ---------------------------------------------------------------------------
# Suite discovery (mirrors run_hecbench.py and run_cublas.py paths)
# ---------------------------------------------------------------------------

TEST_DIR = PROJECT_ROOT / "test"
from reproduce.figures.engines.bench_bundle import discover_kernels, FIG_SUITES




def collect_conflict_pattern_stats(module, ta):
    """Run TypeAnalysis just far enough to collect typed conflict records."""
    stats = {
        "conflicts": 0,
        "conflict_details": [],
    }

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

        stats["conflicts"] += len(ta.conflicts)

        # Harvest detailed conflict records from the monkey-patched log
        if hasattr(ta, '_conflict_log'):
            for c in ta._conflict_log:
                pair, pattern = classify_conflict(c)
                stats["conflict_details"].append({
                    "reg": c.reg or "?",
                    "pair": pair,
                    "pattern": pattern,
                    "def_opcode": c.def_opcode,
                    "use_opcode": c.use_opcode,
                })
            ta._conflict_log.clear()
    return stats


# ---------------------------------------------------------------------------
# Main pipeline: parse → collect stats
# ---------------------------------------------------------------------------

import os as _os
_KERNEL_TIMEOUT = int(_os.environ.get("SLIFTER_EVAL_TIMEOUT", "120"))  # seconds per kernel


def _collect_stats_inner(name, sass_path, elf_path):
    """Inner function — runs in subprocess for timeout enforcement."""
    return collect_stats_for_kernel(name, sass_path, elf_path)


def _per_kernel_result_path(output_dir, suite, kernel_name):
    safe_name = f"{suite}__{kernel_name}".replace("/", "_")
    return output_dir / "per_kernel" / f"{safe_name}.json"


def _legacy_per_kernel_result_path(output_dir, kernel_name):
    safe_name = kernel_name.replace("/", "_")
    return output_dir / "per_kernel" / f"{safe_name}.json"


def _load_cached_kernel_stats(output_dir, suite, kernel_name):
    for path in (
        _per_kernel_result_path(output_dir, suite, kernel_name),
        _legacy_per_kernel_result_path(output_dir, kernel_name),
    ):
        if not path.exists():
            continue
        try:
            with open(path) as f:
                stats = json.load(f)
        except Exception:
            continue
        if isinstance(stats, dict) and stats.get("name") == kernel_name and stats.get("status") == "ok":
            return stats
    return None


def _write_kernel_stats(output_dir, suite, stats):
    result_path = _per_kernel_result_path(output_dir, suite, stats["name"])
    with open(result_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)


def collect_stats_for_kernel(name, sass_path, elf_path):
    """Parse a kernel and collect conflict-only stats."""
    stats = {"name": name, "status": "ok", "error": None}

    try:
        with contextlib.ExitStack() as stack:
            if _os.environ.get("SLIFTER_QUIET_WORKERS", "1") != "0":
                devnull = stack.enter_context(open(_os.devnull, "w"))
                stack.enter_context(contextlib.redirect_stdout(devnull))
                stack.enter_context(contextlib.redirect_stderr(devnull))

            # Parse (mirrors main.py); text-only .elf (cutlass/cuda_sdk) -> Asm.
            sass_text = Path(sass_path).read_text()
            if elf_path is None:
                from utils.asm import Asm
                elf = Asm(sass_text, path=Path(sass_path))
            else:
                elf = Elf(str(elf_path))
            sass_parser = SaSSParser(sass_text, elf=elf)
            module = Module(name, sass_parser)

            # Run transforms up to (but not including) TypeAnalysis
            # Mirrors X86Lifter.get_transform_passes() pipeline
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

            # Pipeline matching X86Lifter up to (but not including) TypeAnalysis
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

            ta2 = TypeAnalysis()
            stats["conflict_analysis"] = collect_conflict_pattern_stats(module, ta2)

    except Exception as e:
        stats["status"] = "error"
        stats["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        stats["traceback"] = traceback.format_exc()[-500:]

    return stats


# ---------------------------------------------------------------------------
# Aggregate and output
# ---------------------------------------------------------------------------

def aggregate_stats(all_stats, suite):
    """Aggregate per-kernel conflict stats into suite-level summaries."""
    summary = {
        "suite": suite,
        "total_kernels": len(all_stats),
        "successful": sum(1 for s in all_stats if s["status"] == "ok"),
        "failed": sum(1 for s in all_stats if s["status"] != "ok"),
    }

    # Conflict pattern analysis
    pattern_counter = Counter()
    pair_counter = Counter()
    domain_pattern = defaultdict(Counter)
    all_conflict_details = []
    conflicts_per_kernel = []
    for s in all_stats:
        conflict_stats = s.get("conflict_analysis", s.get("type_propagation"))
        if not conflict_stats:
            continue
        details = conflict_stats.get("conflict_details", [])
        domain = get_domain(s["name"])
        conflicts_per_kernel.append({
            "kernel": s["name"],
            "domain": domain,
            "conflicts": conflict_stats.get("conflicts", 0),
        })
        for cd in details:
            pattern_counter[cd["pattern"]] += 1
            pair_counter[cd["pair"]] += 1
            domain_pattern[domain][cd["pattern"]] += 1
            all_conflict_details.append({
                "kernel": s["name"], "domain": domain, **cd
            })
    summary["total_conflicts"] = sum(pattern_counter.values())
    summary["conflict_patterns"] = dict(pattern_counter)
    summary["conflict_pairs"] = dict(pair_counter)
    summary["conflict_by_domain"] = {
        d: {"total": sum(c.values()), "patterns": dict(c)}
        for d, c in domain_pattern.items()
    }
    summary["conflict_details"] = all_conflict_details
    summary["conflicts_per_kernel"] = conflicts_per_kernel

    return summary


def write_csv(all_stats, output_dir, suite):
    """Write conflict-centric CSVs."""
    import csv

    # Flat conflict detail rows
    csv_path = output_dir / f"{suite}_conflict_details.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "reg", "pair", "pattern", "def_opcode", "use_opcode"])
        for s in all_stats:
            conflict_stats = s.get("conflict_analysis", s.get("type_propagation"))
            if not conflict_stats:
                continue
            domain = get_domain(s["name"])
            for cd in conflict_stats.get("conflict_details", []):
                w.writerow([
                    s["name"], domain,
                    cd.get("reg", "?"),
                    cd.get("pair", ""),
                    cd.get("pattern", ""),
                    cd.get("def_opcode", ""),
                    cd.get("use_opcode", ""),
                ])

    # Per-kernel conflict totals
    csv_path = output_dir / f"{suite}_conflict_totals.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "conflicts"])
        for s in all_stats:
            conflict_stats = s.get("conflict_analysis", s.get("type_propagation"))
            if not conflict_stats:
                continue
            domain = get_domain(s["name"])
            w.writerow([
                s["name"], domain,
                conflict_stats.get("conflicts", 0),
            ])

    # Domain × pattern breakdown
    domain_pattern = defaultdict(Counter)
    pattern_totals = Counter()
    for s in all_stats:
        conflict_stats = s.get("conflict_analysis", s.get("type_propagation"))
        if not conflict_stats:
            continue
        domain = get_domain(s["name"])
        for cd in conflict_stats.get("conflict_details", []):
            domain_pattern[domain][cd["pattern"]] += 1
            pattern_totals[cd["pattern"]] += 1

    csv_path = output_dir / f"{suite}_conflict_by_domain.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "pattern", "count"])
        for domain in sorted(domain_pattern):
            for pattern, count in sorted(domain_pattern[domain].items(),
                                         key=lambda x: (-x[1], x[0])):
                w.writerow([domain, pattern, count])

    csv_path = output_dir / f"{suite}_conflict_patterns.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pattern", "count"])
        for pattern, count in sorted(pattern_totals.items(), key=lambda x: (-x[1], x[0])):
            w.writerow([pattern, count])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

ALL_SUITES = ["hecbench", "cuobjdump", "ml", "cutlass", "flash_attn",
              "cublas", "cublaslt", "cudnn_cnn", "cudnn_ops", "cudnn_adv",
              "all_sdk"]


def run_suite(suite, output_dir, kernel_filter="", limit=0, parallel=1, force=False):
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

    cached_stats = []
    pending_kernels = []
    for kernel in kernels:
        name = kernel[0]
        cached = None if force else _load_cached_kernel_stats(output_dir, suite, name)
        if cached is None:
            pending_kernels.append(kernel)
        else:
            cached_stats.append(cached)

    all_stats = list(cached_stats)
    if cached_stats:
        print(f"Resuming from {len(cached_stats)} completed kernels; {len(pending_kernels)} remaining")
    if not pending_kernels:
        print("All kernels already processed; rebuilding suite summary from cached results.")

    if parallel > 1 and pending_kernels:
        import multiprocessing
        # Use imap_unordered for progress + per-kernel timeout via maxtasksperchild
        with multiprocessing.Pool(parallel, maxtasksperchild=1) as pool:
            args_list = [(n, s, e) for n, s, e in pending_kernels]
            async_results = []
            for args in args_list:
                ar = pool.apply_async(collect_stats_for_kernel, args)
                async_results.append((args[0], ar))
            for i, (kname, ar) in enumerate(async_results, start=len(cached_stats) + 1):
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
                print(f"  [{i}/{len(kernels)}] {kname:<40s} {tag}")
                # Write per-kernel JSON incrementally
                _write_kernel_stats(output_dir, suite, stats)
    else:
        for i, (name, sass_path, elf_path) in enumerate(pending_kernels, start=len(cached_stats) + 1):
            stats = collect_stats_for_kernel(name, sass_path, elf_path)
            all_stats.append(stats)

            tag = "OK" if stats["status"] == "ok" else f"ERR: {stats.get('error', '')[:60]}"
            print(f"  [{i}/{len(kernels)}] {name:<40s} {tag}")

            _write_kernel_stats(output_dir, suite, stats)

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
    print_summary(summary)

    return suite, all_stats


def print_summary(summary):
    """Print conflict-only summary stats to stdout."""
    print(f"\nConflicts: {summary.get('total_conflicts', 0):,}")

    by_domain = summary.get("conflict_by_domain", {})
    if by_domain:
        print("Top domains:")
        for domain, payload in sorted(by_domain.items(),
                                      key=lambda x: (-x[1].get("total", 0), x[0]))[:8]:
            print(f"  {domain:16s} {payload.get('total', 0):8d}")

    patterns = summary.get("conflict_patterns", {})
    if patterns:
        print("Top patterns:")
        for pattern, count in sorted(patterns.items(), key=lambda x: (-x[1], x[0]))[:12]:
            print(f"  {pattern:35s} {count:8d}")


def main():
    parser = argparse.ArgumentParser(
        description="Collect type-conflict pattern statistics"
    )
    parser.add_argument("--suite", required=True,
                        choices=ALL_SUITES + ["all"],
                        help="Benchmark suite to process ('all' = all suites)")
    parser.add_argument("--output", required=True, help="Output directory for stats")
    parser.add_argument("--kernels", default="", help="Comma-separated kernel name filter")
    parser.add_argument("--limit", type=int, default=0, help="Max kernels per suite (0=all)")
    parser.add_argument("-j", "--parallel", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", action="store_true",
                        help="Ignore cached per-kernel results and rerun the suite")
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
                                          args.limit, args.parallel, args.force)
        if all_stats:
            summary = json.loads((suite_dir / f"{suite}_summary.json").read_text())
            print_summary(summary)
            print(f"\nOutput: {suite_dir}/")

    if len(suites) > 1:
        print(f"\n{'='*60}")
        print(f"All suites complete. Results in {output_dir}/")
        print(f"{'='*60}")

    print("\nOutput files:")
    if len(suites) == 1:
        prefix = output_dir / args.suite
        print(f"  {prefix}_summary.json")
        print(f"  {prefix}_conflict_details.csv")
        print(f"  {prefix}_conflict_totals.csv")
        print(f"  {prefix}_conflict_by_domain.csv")
        print(f"  {prefix}_conflict_patterns.csv")
    else:
        print(f"  {output_dir}/<suite>/<suite>_summary.json")
        print(f"  {output_dir}/<suite>/<suite>_conflict_details.csv")
        print(f"  {output_dir}/<suite>/<suite>_conflict_totals.csv")
        print(f"  {output_dir}/<suite>/<suite>_conflict_by_domain.csv")
        print(f"  {output_dir}/<suite>/<suite>_conflict_patterns.csv")


if __name__ == "__main__":
    main()
