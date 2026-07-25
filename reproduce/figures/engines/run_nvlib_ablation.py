#!/usr/bin/env python3
"""NPP/nvJPEG ablation (paper Figure 8b): five configurations, 98 kernels.

Same mechanism as the HeCBench ablation: committed CPU drivers from
`benchmark/nv_libraries/<kernel>/host_cpu.cpp` and the shared executor in
`kernel_exec.py`. Reports paper labels B0--B4 directly.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from reproduce.figures.engines.kernel_exec import (
    STATUSES, lift_kernel, execute_driver)
from reproduce.figures.engines.manifest_bundle import discover_nvlib_inputs
from reproduce.figures.engines.run_ablation import _ablation_env

# paper labels (Table 8 / Figure 8b) -> internal ablation configs
PAPER_TO_INTERNAL = {"B0": "B0", "B1": "B1", "B2": "B3",
                     "B3": "B2a", "B4": "B2"}


def _library_group(name):
    if name.startswith("npp_compress_"):
        return "npp_compress"
    if name.startswith("npp_uf_"):
        return "npp_uf"
    if name.startswith("npp_"):
        return "npp"
    return "nvjpeg"


def run_one(name, sass, driver, cubin, paper_config, work_root):
    work = Path(work_root) / paper_config / name
    work.mkdir(parents=True, exist_ok=True)
    env = _ablation_env(PAPER_TO_INTERNAL[paper_config])
    ll_path, status, detail = lift_kernel(sass, cubin, work / "kernel",
                                          extra_env=env)
    result = {"name": name, "cubin": _library_group(name),
              "config": paper_config}
    if status != "ok":
        result.update(status=status, detail=detail)
        return result
    nvlib_root = Path(driver).parent.parent
    status, detail = execute_driver(ll_path, driver, [nvlib_root], work)
    result.update(status=status, detail=detail)
    return result


def main():
    ap = argparse.ArgumentParser(description="NPP/nvJPEG ablation (Figure 8b)")
    ap.add_argument("-j", "--parallel", type=int, default=8)
    ap.add_argument("--configs", default="B0,B1,B2,B3,B4")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path)
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = sorted(set(configs) - set(PAPER_TO_INTERNAL))
    if unknown:
        raise SystemExit(f"unknown configurations: {', '.join(unknown)}")

    inputs = discover_nvlib_inputs()
    if args.limit:
        inputs = inputs[:args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    work_root = args.work_dir or args.output / "nvlib_work"
    work_root.mkdir(parents=True, exist_ok=True)

    tasks = [(name, str(sass), str(driver), str(cubin), cfg, str(work_root))
             for name, sass, driver, cubin in inputs for cfg in configs]
    print(f"NPP/nvJPEG ablation: {len(inputs)} kernels x "
          f"{len(configs)} configurations = {len(tasks)} runs")

    results = []
    with ProcessPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, *t): (t[0], t[4]) for t in tasks}
        for done, fut in enumerate(as_completed(futures), 1):
            name, config = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                results.append({"name": name, "cubin": _library_group(name),
                                "config": config, "status": "crash",
                                "detail": f"{type(exc).__name__}: {exc}"})
            if done % 50 == 0 or done == len(tasks):
                print(f"  [{done}/{len(tasks)}]")

    results.sort(key=lambda r: (r["name"], r["config"]))
    out = args.output / "ablation_nvlib_perkernel.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "cubin", "config",
                                           "status", "detail"])
        w.writeheader()
        w.writerows(results)
    for cfg in configs:
        counts = Counter(r["status"] for r in results if r["config"] == cfg)
        print(f"  {cfg}: " + " ".join(f"{s}={counts.get(s, 0)}"
                                      for s in STATUSES))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
