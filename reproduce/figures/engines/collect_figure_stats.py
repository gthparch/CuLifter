#!/usr/bin/env python3
"""Single-pass figure-data collection.

The three original engines (collect_eval_stats, collect_type_details,
collect_conflict_stats) each lifted the entire corpus independently — three
full lift passes over cuBLAS/cuDNN/etc. This collector lifts each kernel ONCE
(applying the FlashAttention kname filter so bwd kernels lift one variant, not
~24) and runs every analysis on that shared lifted module, then writes all the
per-suite CSVs the figures consume. Same numbers, ~3x less work.

Outputs per suite (under $SLIFTER_FIG_DATA, default reproduce/figures/_data):
  per_suite/<suite>_{type_classification,type_propagation,control_flow,
                     propagation_reach}.csv + <suite>_summary.json   (figs 4,6,7)
  type_details/<suite>_type_composition.csv                          (fig 5)
  conflicts/<suite>/<suite>_conflict_by_domain.csv                   (fig 10)
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf
from transform.typeanalysis import TypeAnalysis, TOP, INT32, FLOAT32, INT64, FLOAT64

from reproduce.figures.engines.bench_bundle import (
    filter_to_kname, FIG_SUITES)
if os.environ.get("SLIFTER_FIG_DISCOVERY", "manifest") == "manifest":
    from reproduce.figures.engines.manifest_bundle import discover_kernels
else:
    from reproduce.figures.engines.bench_bundle import discover_kernels
from reproduce.figures.engines.bench_domains import get_domain
from reproduce.figures.engines import collect_eval_stats as E
from reproduce.figures.engines.collect_type_details import TYPE_NAMES, FP_OPCODES
from reproduce.figures.engines.collect_conflict_stats import collect_conflict_pattern_stats

_KERNEL_TIMEOUT = int(os.environ.get("SLIFTER_EVAL_TIMEOUT", "600"))
DATA = Path(os.environ.get("SLIFTER_FIG_DATA", str(PROJECT_ROOT / "reproduce/figures/_data")))


def _make_passes():
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
    return [
        BuildCFG(), ResolveBRX(), HandleCall(), CFGSimplify(),
        CGPatterns(), WarpPatterns(), SRSubstitute(), OpModTransform(),
        PredicatedToSel(), SSA(), RegRemap(), DefUseAnalysis(),
        XmadToImad(), FPHack(), RegRemap(), SetZero(),
        DefUseAnalysis(), MovEliminate(), DefUseAnalysis(), DCE(),
        CFGSimplify(), DefUseAnalysis(), RegRemap(), DefUseAnalysis(),
        DCE(), CFGSimplify(), DefUseAnalysis(),
    ]


def _composition_from_lifted(module):
    """Type composition (T1) + unresolved-consumer (T2) — extracted verbatim
    from collect_type_details.collect_type_details_for_kernel so fig5 numbers
    are unchanged."""
    ta = TypeAnalysis()
    ta.type_stats = {"unique_resolved": 0, "ambiguous_int32": 0,
                     "ambiguous_int64": 0, "ambiguous_other": 0,
                     "empty_default": 0, "reg_group": 0, "total": 0}
    ta.type_detail_counts = {}
    type_counts = Counter()
    unresolved_consumers = Counter()

    for func in module.functions:
        ta.conflicts = set()
        work_list = ta.traverse_cfg(func)
        optypes = {}
        for bb in work_list:
            for inst in bb.instructions:
                ta.resolve_types(inst, optypes)
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
        for bb in work_list:
            for inst in bb.instructions:
                for op in inst.operands:
                    if op.is_reg_group:
                        for child_op in op.reg_group:
                            child_op.type_desc = ta.get_type_desc(child_op, optypes)
                    op.type_desc = ta.get_type_desc(op, optypes)
        for key, val in optypes.items():
            if val == TOP or len(val) >= len(TOP):
                type_counts["unresolved"] += 1
            elif len(val) == 1:
                type_counts[TYPE_NAMES.get(val, str(val))] += 1
            elif val == INT32 | FLOAT32:
                type_counts["Num32"] += 1
            elif val == INT64 | FLOAT64:
                type_counts["Num64"] += 1
            else:
                type_counts["other_set"] += 1
        wide_groups = {}
        for bb in work_list:
            for inst in bb.instructions:
                for op in inst.operands:
                    if not op.is_reg_group:
                        continue
                    gk = op.get_ir_name(None)
                    if not gk:
                        continue
                    wide_groups[gk] = {"group_type": op.get_type_desc(),
                                       "child_types": [c.get_type_desc() for c in op.reg_group]}
        for payload in wide_groups.values():
            gt = payload["group_type"]
            if gt in ("Int64", "Float64"):
                continue
            for ct in payload["child_types"]:
                if type_counts.get(ct, 0) > 0:
                    type_counts[ct] -= 1
            type_counts[gt] += 1
        unresolved_ops = {k for k, v in optypes.items()
                          if v == TOP or len(v) >= len(TOP)}
        for bb in work_list:
            for inst in bb.instructions:
                opcode = inst.opcodes[0]
                for use_op in inst.get_uses():
                    if use_op in unresolved_ops:
                        if opcode in FP_OPCODES or opcode[:1] in ("F", "H", "D"):
                            unresolved_consumers["fp_consumer"] += 1
                        else:
                            unresolved_consumers["int_consumer"] += 1
    return dict(type_counts), dict(unresolved_consumers)


def collect_all_for_kernel(name, sass_path, elf_path):
    """Lift once, run every analysis. Returns a stats dict carrying the union of
    what the three engines produced (so the existing writers/exporters apply)."""
    stats = {"name": name, "status": "ok", "error": None}
    try:
        sass_text = Path(sass_path).read_text()
        if elf_path is None:
            from utils.asm import Asm
            elf = Asm(sass_text, path=Path(sass_path))
        else:
            elf = Elf(str(elf_path))
        module = Module(name, SaSSParser(sass_text, elf=elf))
        filter_to_kname(module, sass_path)   # FA: keep only the launched kernel

        # control flow on the raw module (before transforms), as eval does
        stats["control_flow"] = E.collect_control_flow_stats(module)

        # lift: run the pipeline ONCE (per-function for multi-function modules)
        from types import SimpleNamespace
        all_funcs = list(module.functions)
        if len(all_funcs) > 1:
            transformed = []
            for func in all_funcs:
                single = SimpleNamespace(name=module.name, isa=module.isa,
                                         elf=module.elf, functions=[func])
                for t in _make_passes():
                    t.apply(single)
                transformed.append(single.functions[0])
            module.functions = transformed
        else:
            for t in _make_passes():
                t.apply(module)

        # All analyses share the single lifted module (each runs its own ta,
        # exactly as in the standalone engines). Each is isolated so that one
        # analysis failing on a kernel does not drop the others — this matches
        # the original three-engine behavior, where each engine succeeded or
        # failed per kernel independently (e.g. composition's set-key path can
        # fail on some flash/cuBLAS kernels while classification still succeeds).
        def _try(key, fn):
            try:
                return fn()
            except Exception as ex:
                stats.setdefault("analysis_errors", {})[key] = f"{type(ex).__name__}: {str(ex)[:120]}"
                return None
        tc = _try("type_classification", lambda: E.classify_instructions(module, TypeAnalysis()))
        if tc is not None:
            stats["type_classification"] = tc
        tp = _try("type_propagation", lambda: E.collect_type_propagation_stats(module, TypeAnalysis()))
        if tp is not None:
            stats["type_propagation"] = tp
        comp = _try("type_composition", lambda: _composition_from_lifted(module))
        if comp is not None:
            stats["type_composition"], stats["unresolved_consumers"] = comp
        cf = _try("conflicts", lambda: collect_conflict_pattern_stats(module, TypeAnalysis()))
        if cf is not None:
            stats["conflicts_detail"] = cf
    except Exception as e:
        import traceback
        stats["status"] = "error"
        stats["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        stats["traceback"] = traceback.format_exc()[-500:]
    return stats


def _write_composition(all_stats, out_dir, suite):
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = set()
    for r in all_stats:
        cols.update(r.get("type_composition", {}).keys())
    cols = sorted(cols)
    with open(out_dir / f"{suite}_type_composition.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kernel", "domain", "suite"] + cols)
        for r in all_stats:
            if "type_composition" not in r:
                continue
            tc = r["type_composition"]
            w.writerow([r["name"], get_domain(r["name"]), suite] + [tc.get(c, 0) for c in cols])


def _write_conflict_by_domain(all_stats, out_dir, suite):
    out_dir.mkdir(parents=True, exist_ok=True)
    by = Counter()   # (domain, pattern) -> count
    for r in all_stats:
        dom = get_domain(r["name"])
        for d in r.get("conflicts_detail", {}).get("conflict_details", []):
            by[(dom, d.get("pattern", "unclassified"))] += 1
    with open(out_dir / f"{suite}_conflict_by_domain.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["domain", "pattern", "count"])
        for (dom, pat), c in sorted(by.items()):
            w.writerow([dom, pat, c])


def run_suite(suite, parallel=1, limit=0):
    kernels = discover_kernels(suite)
    if limit:
        kernels = kernels[:limit]
    if not kernels:
        print(f"  {suite}: no kernels")
        return suite, []
    print(f"  {suite}: {len(kernels)} kernels (single pass)")
    all_stats = []
    if parallel > 1:
        import multiprocessing
        with multiprocessing.Pool(parallel, maxtasksperchild=1) as pool:
            ars = [(n, pool.apply_async(collect_all_for_kernel, (n, s, e)))
                   for n, s, e in kernels]
            for n, ar in ars:
                try:
                    all_stats.append(ar.get(timeout=_KERNEL_TIMEOUT))
                except Exception as exc:
                    all_stats.append({"name": n, "status": "error", "error": str(exc)[:200]})
    else:
        for n, s, e in kernels:
            all_stats.append(collect_all_for_kernel(n, s, e))

    per_suite = DATA / "per_suite"
    per_suite.mkdir(parents=True, exist_ok=True)
    (DATA / "results" / suite).mkdir(parents=True, exist_ok=True)
    # eval CSVs + summary.json (figs 4,6,7) — reuse the tested writer
    E.write_csv(all_stats, per_suite, suite)
    summary = E.aggregate_stats(all_stats, suite)
    with open(per_suite / f"{suite}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # copy summary where fig7b/results expects it too
    with open(DATA / "results" / suite / f"{suite}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    _write_composition(all_stats, DATA / "type_details", suite)             # fig5
    _write_conflict_by_domain(all_stats, DATA / "conflicts" / suite, suite)  # fig10
    ok = sum(1 for s in all_stats if s["status"] == "ok")
    print(f"  {suite}: {ok}/{len(all_stats)} analyzed")
    return suite, all_stats


def main():
    ap = argparse.ArgumentParser(description="Single-pass figure-data collection")
    ap.add_argument("--suite", default="all")
    ap.add_argument("-j", "--parallel", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    suites = FIG_SUITES if args.suite == "all" else [args.suite]
    for s in suites:
        run_suite(s, parallel=args.parallel, limit=args.limit)
    # Stamp completeness only for a full, unlimited sweep of every suite, so a
    # partial (--limit / single-suite) run can never be mistaken for complete
    # by ensure_all_stats.
    if args.suite == "all" and not args.limit:
        (DATA / ".collection_complete").write_text(
            "suites=%s\n" % ",".join(suites))
        print("stamped .collection_complete")


if __name__ == "__main__":
    main()
