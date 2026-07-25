#!/usr/bin/env python3
"""Type-recovery-step ablation on HeCBench (paper Figure 8a / Table 8).

Runs the five ablation configurations over the 229 HeCBench units of the
benchmark bundle. Every unit uses its committed CPU driver
(`benchmark/hecbench/<name>/host_cpu.cpp`, produced by the HeCBench
end-to-end lane) and the shared lift/compile/link/run executor
(`kernel_exec.py`), the same mechanism as the NPP/nvJPEG ablation.
Benchmarks that ship no device code (empty SASS) have nothing to ablate and
are recorded as pass.

The CSV exporter maps the internal configuration keys (B0, B1, B2, B2a, B3)
to the paper's B0--B4 labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.figures.engines.kernel_exec import (
    STATUSES, lift_kernel, execute_driver)

BENCHMARK_DIR = Path(os.environ.get("SLIFTER_BENCHMARK_DIR",
                                    PROJECT_ROOT / "benchmark"))
HECBENCH_DIR = BENCHMARK_DIR / "hecbench"
_MIN_SASS = 200   # bytes; below this the benchmark ships no device code
_HINTS_FILE = PROJECT_ROOT / "scripts" / "hecbench" / "driver_hints.json"


def _lockstep_benches():
    """Benchmarks whose tracked contract requires warp-lockstep emulation
    (pre-Volta warp-synchronous shared memory), same source of truth as the
    HeCBench end-to-end lane."""
    try:
        import json as _json
        hints = _json.loads(_HINTS_FILE.read_text())
    except Exception:
        return set()
    return {k for k, v in hints.items()
            if isinstance(v, dict) and v.get("lockstep")}


LOCKSTEP = _lockstep_benches()

ABLATION_SKIP = {
    "B0": set(),                 # full pipeline
    "B1": {"PredicatedToSel"},   # no predicate->SEL
    "B2": {"TypeAnalysis"},      # all-Int32 (no typing)
    "B2a": {"TypeAnalysis"},     # seed only (no propagation)
    "B3": {"XmadToImad", "FPHack", "SetZero", "MovEliminate"},  # no pattern norm
}
ABLATION_LABELS = {
    "B0": "Full pipeline",
    "B1": "No predicate→SEL",
    "B2": "All-Int32 (no typing)",
    "B2a": "Seed-only (no propagation)",
    "B3": "No pattern normalization",
}


def _ablation_env(config):
    env = {}
    skips = ABLATION_SKIP.get(config, set())
    if skips:
        env["SLIFTER_SKIP_PASSES"] = ",".join(skips)
    if config == "B2a":
        env["SLIFTER_TYPE_SEED_ONLY"] = "1"
    return env


def discover_units(limit=0):
    """(name, sass, cubin, driver | None, has_device_code) for every unit."""
    units = []
    for sass in sorted(HECBENCH_DIR.glob("*.sass")):
        name = sass.stem
        cubin = HECBENCH_DIR / "cubins" / f"{name}.cubin"
        if not cubin.is_file():
            continue
        has_code = sass.stat().st_size > _MIN_SASS
        driver = HECBENCH_DIR / name / "host_cpu.cpp"
        units.append((name, sass, cubin, driver if driver.is_file() else None,
                      has_code))
    if limit:
        units = units[:limit]
    return units


def run_one(name, sass, cubin, driver, config, work_root):
    work = Path(work_root) / config / name
    work.mkdir(parents=True, exist_ok=True)
    lift_env = _ablation_env(config)
    exec_env = None
    if name in LOCKSTEP:
        lift_env = {**lift_env, "CULIFTER_LOCKSTEP": "1"}
        exec_env = {"SLIFTER_LOCKSTEP": "1"}
    ll_path, status, detail = lift_kernel(
        sass, cubin, work / "kernel", extra_env=lift_env)
    if status != "ok":
        return {"name": name, "config": config, "status": status,
                "detail": detail}
    status, detail = execute_driver(
        ll_path, driver, [HECBENCH_DIR / name], work, exec_env=exec_env)
    return {"name": name, "config": config, "status": status,
            "detail": detail}


def main():
    ap = argparse.ArgumentParser(description="HeCBench ablation (Figure 8a)")
    ap.add_argument("-j", "--parallel", type=int, default=6)
    ap.add_argument("--configs", default="B0,B1,B2,B2a,B3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", default="reproduce/figures/_data/ablation")
    ap.add_argument("--work-dir", default=None)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = sorted(set(configs) - set(ABLATION_SKIP))
    if unknown:
        raise SystemExit(f"unknown configurations: {', '.join(unknown)}")

    units = discover_units(args.limit)
    if not units:
        raise SystemExit(f"no HeCBench units under {HECBENCH_DIR}")
    missing = [n for n, _, _, d, code in units if code and d is None]
    if missing:
        raise SystemExit(
            f"{len(missing)} unit(s) lack benchmark/hecbench/<name>/host_cpu.cpp "
            f"(e.g. {missing[:5]}); install a bundle that ships the drivers")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir or PROJECT_ROOT / "temp" / "ablation")
    work_root.mkdir(parents=True, exist_ok=True)

    runnable = [(n, s, c, d) for n, s, c, d, code in units if code]
    trivial = [n for n, _, _, _, code in units if not code]
    tasks = [(n, str(s), str(c), str(d), cfg, str(work_root))
             for n, s, c, d in runnable for cfg in configs]
    print(f"Ablation: {len(units)} units ({len(trivial)} without device code) "
          f"x {len(configs)} configs = {len(tasks)} executed runs")

    results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, *t): (t[0], t[4]) for t in tasks}
        for done, fut in enumerate(as_completed(futures), 1):
            name, config = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"name": name, "config": config,
                                "status": "crash",
                                "detail": f"{type(exc).__name__}: {exc}"})
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]")
    for cfg in configs:
        for name in trivial:
            results.append({"name": name, "config": cfg, "status": "pass",
                            "detail": "no device code"})

    results.sort(key=lambda r: (r["name"], r["config"]))
    with open(out_dir / "ablation_results.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "config", "status",
                                           "detail"])
        w.writeheader()
        w.writerows(results)

    summary = {}
    for cfg in configs:
        counts = Counter(r["status"] for r in results if r["config"] == cfg)
        total = sum(counts.values())
        summary[cfg] = {
            "label": ABLATION_LABELS[cfg],
            "total": total,
            **{s: counts.get(s, 0) for s in STATUSES},
            "pass_rate": round(100.0 * counts.get("pass", 0) / max(total, 1), 1),
        }
        print(f"  {cfg}: " + " ".join(f"{s}={counts.get(s, 0)}"
                                      for s in STATUSES))
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Results: {out_dir / 'ablation_results.csv'}")
    print(f"Summary: {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
