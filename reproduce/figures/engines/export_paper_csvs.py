#!/usr/bin/env python3
"""Aggregate the per-suite characterization data into the paper's CSVs.

Reads the per-suite measurement CSVs written by collect_figure_stats.py
(_data/per_suite, _data/type_details, _data/conflicts) — all products of ONE
lift pass over the manifest corpus — and emits the plot-ready CSVs under the
same names the paper source uses (figs/csv/fig5_*.csv ... fig11_*.csv, i.e.
paper Figures 4-10 in the draft numbering the source kept).

Instruction and value sums cover every function in the corpus (kernels
plus device helpers). `n_kernels` counts exported kernels, one per unit for
hecbench/cutlass/cuda_sdk and the suite totals for the vendor-library and
FlashAttention suites, whose units are modules or selected kernels.
fig_population.json records both accountings for the run.
"""
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.figures.engines.manifest_bundle import PAPER_KERNELS, SUITE_KEY

DATA = Path(os.environ.get("SLIFTER_FIG_DATA", str(PROJECT_ROOT / "reproduce/figures/_data")))
OUT = Path(os.environ.get("SLIFTER_FIG_OUT", str(PROJECT_ROOT / "reproduce/figures/output")))
PER_SUITE = DATA / "per_suite"
TYPE_DETAILS = DATA / "type_details"
CONFLICTS = DATA / "conflicts"

FIG_SUITES = list(SUITE_KEY)

DOMAIN_ORDER = [
    "Linear Algebra", "ML/DNN", "Simulation", "Graph", "Image/Vision",
    "Crypto", "Science", "Sort/Search", "Stencil/PDE", "Infrastructure",
    "Other",
]
# suites whose unit count is NOT the paper kernel count
SUITE_LEVEL_KERNELS = {"cublas", "cudnn_cnn", "cudnn_ops", "cudnn_adv", "flash_attn"}

TAXONOMY_PATTERNS = [
    "ieee_754_field_ops", "float_reconstruction", "fast_math_chains",
    "branch_merge_phi", "const_materialization", "unclassified",
]
COMPOSITION_ORDER = ["Bool", "Float32", "Int32", "Int64", "Num32",
                     "unresolved", "Float16", "other_set", "Float64", "Int128"]


def rows_of(path):
    if not path.is_file():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def order_domains(domains):
    known = [d for d in DOMAIN_ORDER if d in domains]
    return known + sorted(d for d in domains if d not in DOMAIN_ORDER)


def kernels_by_domain():
    """Map domain -> paper kernel count, using unit domains from the
    classification CSVs (1 kernel/unit) and suite constants for module suites."""
    kern = Counter()
    units = Counter()
    notes = {}
    for suite in FIG_SUITES:
        suite_rows = rows_of(PER_SUITE / f"{suite}_type_classification.csv")
        dom_units = Counter(r["domain"] for r in suite_rows)
        for d, n in dom_units.items():
            units[d] += n
        if suite in SUITE_LEVEL_KERNELS:
            if len(dom_units) == 1:
                only = next(iter(dom_units))
                kern[only] += PAPER_KERNELS[suite]
            elif dom_units:
                # keep deterministic: all suite kernels on the majority domain
                major = dom_units.most_common(1)[0][0]
                kern[major] += PAPER_KERNELS[suite]
                notes[suite] = f"multi-domain unit mapping {dict(dom_units)}; " \
                               f"{PAPER_KERNELS[suite]} kernels attributed to {major}"
            else:
                notes[suite] = "no analyzed units"
        else:
            for d, n in dom_units.items():
                kern[d] += n
            missing = PAPER_KERNELS[suite] - sum(dom_units.values())
            if missing:
                notes[suite] = f"{missing} of {PAPER_KERNELS[suite]} paper kernels " \
                               "missing from analyzed units"
    return kern, units, notes


def export_classification(kern):
    agg = defaultdict(lambda: Counter())
    for suite in FIG_SUITES:
        for r in rows_of(PER_SUITE / f"{suite}_type_classification.csv"):
            a = agg[r["domain"]]
            for k in ("seeding", "transparent", "unclassified", "total"):
                a[k] += int(r[k])
    out = OUT / "fig5_type_classification.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "seeding", "transparent", "unclassified", "total",
                    "n_kernels", "seeding_pct", "transparent_pct", "unclassified_pct"])
        for d in order_domains(agg):
            a = agg[d]
            t = a["total"] or 1
            w.writerow([d, a["seeding"], a["transparent"], a["unclassified"],
                        a["total"], kern.get(d, 0),
                        100.0 * a["seeding"] / t, 100.0 * a["transparent"] / t,
                        100.0 * a["unclassified"] / t])
    return out


def export_composition(kern):
    agg = defaultdict(Counter)
    cols = set()
    for suite in FIG_SUITES:
        for r in rows_of(TYPE_DETAILS / f"{suite}_type_composition.csv"):
            d = r["domain"]
            for k, v in r.items():
                if k in ("kernel", "domain", "suite"):
                    continue
                try:
                    val = int(float(v or 0))
                except ValueError:
                    continue
                # raw set-repr keys (e.g. frozenset({'BF16'})) are the
                # mechanism's catch-all bucket; fold as other_set like the
                # published pipeline did
                if k.startswith("frozenset"):
                    k = "other_set"
                agg[d][k] += val
                cols.add(k)
    ordered_cols = [c for c in COMPOSITION_ORDER if c in cols] + \
                   sorted(c for c in cols if c not in COMPOSITION_ORDER)
    out = OUT / "fig6_type_composition.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain"] + ordered_cols + ["total_values", "n_kernels"] +
                   [f"{c}_pct" for c in ordered_cols])
        for d in order_domains(agg):
            a = agg[d]
            total = sum(a[c] for c in ordered_cols) or 1
            w.writerow([d] + [a[c] for c in ordered_cols] +
                       [total, kern.get(d, 0)] +
                       [100.0 * a[c] / total for c in ordered_cols])
    return out


def export_resolution(kern):
    agg = defaultdict(Counter)
    for suite in FIG_SUITES:
        for r in rows_of(PER_SUITE / f"{suite}_type_propagation.csv"):
            a = agg[r["domain"]]
            for k in ("seeded", "propagated", "unresolved", "conflicts", "iterations"):
                a[k] += int(r[k])
    out = OUT / "fig7_type_resolution.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "seeded", "propagated", "unresolved", "conflicts",
                    "iterations", "n_kernels", "total_values",
                    "seeded_pct", "propagated_pct", "unresolved_pct"])
        for d in order_domains(agg):
            a = agg[d]
            total = a["seeded"] + a["propagated"] + a["unresolved"]
            t = total or 1
            w.writerow([d, a["seeded"], a["propagated"], a["unresolved"],
                        a["conflicts"], a["iterations"], kern.get(d, 0), total,
                        100.0 * a["seeded"] / t, 100.0 * a["propagated"] / t,
                        100.0 * a["unresolved"] / t])
    return out


def export_reach():
    hops = Counter()
    for suite in FIG_SUITES:
        for r in rows_of(PER_SUITE / f"{suite}_propagation_reach.csv"):
            hops[int(r["hops"])] += int(r["count"])
    total = sum(hops.values()) or 1
    out = OUT / "fig8a_propagation_reach.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hops", "count", "pct"])
        for h in sorted(hops):
            w.writerow([h, hops[h], 100.0 * hops[h] / total])
    return out


def export_convergence():
    dist90 = Counter()
    for suite in FIG_SUITES:
        p = PER_SUITE / f"{suite}_summary.json"
        if not p.is_file():
            continue
        with open(p) as fh:
            s = json.load(fh)
        for k, v in s.get("convergence", {}).get("distribution", {}).items():
            dist90[int(k)] += v
    out = OUT / "fig8b_convergence.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iterations", "kernels"])
        for it in range(0, max([*dist90, 6]) + 1):
            w.writerow([it, dist90.get(it, 0)])
    return out


# fine-grained conflict pattern -> paper category (from fig10_type_boundary.sh,
# the published Figure 10 aggregation)
TAXONOMY_GROUPS = {
    "ieee_754_field_ops": [
        "exponent_extraction", "mantissa_extraction", "sign_extraction",
        "sign_injection", "exponent_mask_check", "bit_field_extraction",
        "comparison_on_float_bits"],
    "float_reconstruction": [
        "float_reconstruction_1.0", "inf_construction", "exponent_bias_adjust",
        "bitfield_float_construction", "float_reconstruction_bitwise",
        "int_arith_to_float_reinterpret"],
    "fast_math_chains": [
        "rsqrt_range_check", "fast_division_chain_end", "fast_division_bias",
        "int_arith_on_float_bits"],
    "branch_merge_phi": ["phi_merge_type_conflict"],
    "const_materialization": ["float_const_materialization"],
    "unclassified": [
        "unclassified_i32_to_f32", "unclassified_f32_to_i32", "unclassified",
        "width_reinterpret_from_wide", "bitwise_on_float", "int_to_predicate",
        "predicate_to_int", "float_to_predicate", "double_int_reinterpret",
        "int_to_double_reinterpret"],
}
_PAT2GRP = {p: g for g, ps in TAXONOMY_GROUPS.items() for p in ps}


def export_taxonomy():
    agg = defaultdict(Counter)
    unmapped = set()
    for suite in FIG_SUITES:
        for r in rows_of(CONFLICTS / suite / f"{suite}_conflict_by_domain.csv"):
            grp = _PAT2GRP.get(r["pattern"])
            if grp is None:
                unmapped.add(r["pattern"])
                grp = "unclassified"
            agg[r["domain"]][grp] += int(r["count"])
    if unmapped:
        print(f"  [taxonomy] {len(unmapped)} unmapped pattern(s) -> unclassified: "
              f"{sorted(unmapped)}")
    patterns = TAXONOMY_PATTERNS
    out = OUT / "fig11_conflict_taxonomy.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["domain_order", "domain", "total_conflicts"]
        for p in patterns:
            header += [f"{p}_count", f"{p}_pct"]
        w.writerow(header)
        for i, d in enumerate(order_domains(agg), 1):
            a = agg[d]
            total = sum(a.values())
            row = [i, d, total]
            for p in patterns:
                row += [a.get(p, 0), round(100.0 * a.get(p, 0) / total, 4) if total else 0.0]
            w.writerow(row)
    return out


# internal ablation config -> paper label (Table 8 / Figure 8)
INTERNAL_TO_PAPER = [
    ("B0", "B0", "B0 full"),
    ("B1", "B1", "B1 no Pred->SEL"),
    ("B3", "B2", "B2 no pattern norm"),
    ("B2a", "B3", "B3 seed only"),
    ("B2", "B4", "B4 no type recovery"),
]
ABL_STATUSES = ["pass", "wrong_result", "crash", "lift_fail",
                "compile_fail", "link_fail", "timeout"]


def export_ablation():
    p = DATA / "ablation" / "ablation_summary.json"
    if not p.is_file():
        return None
    with open(p) as fh:
        summary = json.load(fh)
    # run_ablation uses lift_timeout as a distinct status; fold into timeout
    out = OUT / "fig9_ablation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "label", "total", "pass", "pass_rate_pct"]
                   + ABL_STATUSES[1:])
        for internal, paper, label in INTERNAL_TO_PAPER:
            d = summary.get(internal)
            if not d:
                continue
            total = d.get("total", 0)
            timeout = d.get("timeout", 0) + d.get("lift_timeout", 0)
            row = [paper, label, total, d.get("pass", 0),
                   round(100.0 * d.get("pass", 0) / max(total, 1), 1)]
            for s in ABL_STATUSES[1:]:
                row.append(timeout if s == "timeout" else d.get(s, 0))
            w.writerow(row)
    return out


def export_ablation_perdomain():
    p = DATA / "ablation" / "ablation_results.csv"
    if not p.is_file():
        return None
    sys.path.insert(0, str(PROJECT_ROOT))
    from reproduce.figures.engines.bench_domains import get_domain
    lift_ok = Counter()
    exec_ok = Counter()
    total = Counter()
    for r in rows_of(p):
        if r["config"] != "B0":
            continue
        d = get_domain(r["name"])
        total[d] += 1
        if r["status"] not in ("lift_fail", "lift_timeout"):
            lift_ok[d] += 1
        if r["status"] == "pass":
            exec_ok[d] += 1
    out = OUT / "fig9_ablation_perdomain.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["domain", "total_kernels", "successful_lift",
                    "execution_correct_pass", "lift_rate_pct",
                    "correct_rate_pct"])
        for d in order_domains(total):
            t = total[d]
            w.writerow([d, t, lift_ok[d], exec_ok[d],
                        round(100.0 * lift_ok[d] / t, 2),
                        round(100.0 * exec_ok[d] / t, 2)])
    return out


def export_ablation_nvlibs():
    p = DATA / "ablation_nvlib" / "ablation_nvlib_perkernel.csv"
    if not p.is_file():
        return None
    per = defaultdict(Counter)
    for r in rows_of(p):
        per[r["config"]][r["status"]] += 1
    label = {"B0": "B0 full", "B1": "B1 no Pred->SEL",
             "B2": "B2 no pattern norm", "B3": "B3 seed only",
             "B4": "B4 no type recovery"}
    out = OUT / "fig9_ablation_nvlibs.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "label", "total", "pass", "pass_rate_pct"]
                   + ABL_STATUSES[1:])
        for cfg in ("B0", "B1", "B2", "B3", "B4"):
            c = per.get(cfg)
            if c is None:
                continue
            total = sum(c.values())
            w.writerow([cfg, label[cfg], total, c.get("pass", 0),
                        round(100.0 * c.get("pass", 0) / max(total, 1), 1)]
                       + [c.get(s, 0) for s in ABL_STATUSES[1:]])
    return out


def export_degradation():
    p = DATA / "degradation" / "degradation_summary.json"
    if not p.is_file():
        return None
    with open(p) as fh:
        summary = json.load(fh)
    out = OUT / "fig10_type_degradation.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["removal_pct", "total", "lift_ok", "lift_rate",
                    "avg_resolution", "failed_lift"])
        for key in sorted(summary, key=int):
            r = summary[key]
            w.writerow([r["removal_pct"], r["total"], r["lift_ok"],
                        r["lift_rate"], r["avg_resolution"],
                        r["total"] - r["lift_ok"]])
    return out


def main():
    missing = [s for s in FIG_SUITES
               if not (PER_SUITE / f"{s}_type_classification.csv").is_file()]
    if missing:
        raise SystemExit(
            f"missing per-suite characterization data for: {', '.join(missing)} "
            "(run the collect stage first)")
    OUT.mkdir(parents=True, exist_ok=True)
    kern, units, notes = kernels_by_domain()
    outputs = [
        export_classification(kern),
        export_composition(kern),
        export_resolution(kern),
        export_reach(),
        export_convergence(),
        export_taxonomy(),
        export_ablation(),
        export_ablation_perdomain(),
        export_ablation_nvlibs(),
        export_degradation(),
    ]
    outputs = [o for o in outputs if o]
    pop = {
        "paper_kernels_by_domain": dict(kern),
        "paper_kernels_total": sum(kern.values()),
        "lift_units_by_domain": dict(units),
        "lift_units_total": sum(units.values()),
        "notes": notes,
    }
    with open(OUT / "fig_population.json", "w") as fh:
        json.dump(pop, fh, indent=2)
    print(f"population: {pop['paper_kernels_total']} paper kernels over "
          f"{pop['lift_units_total']} lift units")
    for note in notes.values():
        print(f"  note: {note}")
    for o in outputs:
        print(f"wrote {o}")


if __name__ == "__main__":
    main()
