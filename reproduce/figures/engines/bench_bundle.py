#!/usr/bin/env python3
"""Kernel discovery over the download-only `benchmark/` bundle.

Locates each suite's SASS and binary in the layout the bundle ships (one dir
per suite under `benchmark/`), so the figure engines read the same corpus that
`reproduce/run_cpu_sm90.sh` lifts. The open suites need no CUDA toolkit; the
closed library suites (cublas/cudnn) need a CUDA-13.x `cuobjdump` (set
$CUOBJDUMP, or place one at benchmark/common/tools/cuobjdump) to dump SASS from
their module cubins.

Returns, per suite, a list of (name, sass_path, elf_path):
  * elf_path is a binary cubin for hecbench/flash_attn/cublas/cudnn,
  * elf_path is None for cutlass/cuda_sdk (their .elf are nvdisasm TEXT dumps,
    lifted as a single input — the engines pass them as `Asm`, not `Elf`).

`name` carries the suite prefix for the library/cutlass/flash suites
(e.g. "cublas/<module>") so bench_domains.get_domain() maps them to the right
paper domain (cuBLAS→Linear Algebra, cuDNN/CUTLASS/Flash→ML/DNN).
"""
import os
import subprocess
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]           # engines/ -> figures -> reproduce -> repo
BUNDLE = Path(os.environ.get("SLIFTER_BENCHMARK_DIR", _REPO / "benchmark"))
SASS_CACHE = Path(os.environ.get("SLIFTER_FIG_SASS_CACHE",
                                 _REPO / "reproduce" / "figures" / "_data" / "sass_cache"))

# suites the figure pipeline covers (paper Table 6, minus per-kernel splitting)
FIG_SUITES = ["hecbench", "cuobjdump", "cublas",
              "cudnn_cnn", "cudnn_ops", "cudnn_adv", "flash_attn", "cutlass"]

_MIN_SASS = 200   # bytes; smaller means a failed/stub disassembly


def resolve_cuobjdump():
    c = os.environ.get("CUOBJDUMP")
    if c and Path(c).exists():
        return c
    bundled = BUNDLE / "common" / "tools" / "cuobjdump"
    if bundled.exists():
        return str(bundled)
    return "cuobjdump"


def _dump_sass(cubin: Path, out: Path):
    """Disassemble `cubin` to `out` (cached). Returns out or None if empty."""
    if out.exists() and out.stat().st_size > _MIN_SASS:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        try:
            subprocess.run([resolve_cuobjdump(), "--dump-sass", str(cubin)],
                           stdout=f, stderr=subprocess.DEVNULL, timeout=600)
        except Exception:
            return None
    return out if out.exists() and out.stat().st_size > _MIN_SASS else None


def filter_to_kname(module, sass_path):
    """If a kname.txt sits beside the kernel's SASS (FlashAttention dirs), keep
    only that launched kernel's function — the module cubin holds up to ~24
    template variants, and analyzing them all is a ~73 GB / 53M-line blowup for
    the bwd kernels. Mirrors main.py's -name. No-op for suites without kname.txt."""
    kn = Path(sass_path).parent / "kname.txt"
    if not kn.exists():
        return module
    want = kn.read_text().strip()
    if not want:
        return module
    keep = [f for f in module.functions if f.name == want]
    if keep:
        module.functions = keep
    return module


def discover_kernels(suite: str, kernel_filter: str = "", output_dir=None):
    """Discover (name, sass_path, elf_path) for one suite from the bundle."""
    out = []
    B = BUNDLE

    if suite == "hecbench":
        d = B / "hecbench"
        for sass in sorted(d.glob("*.sass")):
            if sass.stat().st_size <= _MIN_SASS:
                continue
            cub = d / "cubins" / f"{sass.stem}.cubin"
            if cub.exists():
                out.append((sass.stem, sass, cub))

    elif suite == "cuobjdump":                 # paper "CUDA SDK"
        for elf in sorted((B / "cuda_sdk" / "elf").glob("*.elf")):
            out.append((elf.stem, elf, None))  # text dump; lifted as a single input

    elif suite == "cutlass":
        for elf in sorted((B / "cutlass" / "elf").glob("*.elf")):
            out.append((f"cutlass/{elf.stem}", elf, None))

    elif suite == "flash_attn":
        d = B / "flash_attention"
        if d.exists():
            for kd in sorted(p for p in d.iterdir() if p.is_dir()):
                cub = kd / "kernel.cubin"
                if not cub.exists():
                    continue
                sass = kd / "kernel.sass"
                if not sass.exists() or sass.stat().st_size <= _MIN_SASS:
                    sass = _dump_sass(cub, SASS_CACHE / "flash_attn" / f"{kd.name}.sass")
                if sass:
                    out.append((kd.name, sass, cub))   # name flash_* -> ML/DNN

    elif suite in ("cublas", "cudnn_cnn", "cudnn_ops", "cudnn_adv"):
        for cub in sorted((B / suite / "cubins").glob("*.cubin")):
            # The bundle ships pre-disassembled SASS beside each module cubin
            # (benchmark/<suite>/<cubin>.sass), so no CUDA cuobjdump is needed.
            # A present-but-tiny .sass is a genuinely empty (PTX-only) module —
            # trust it and skip, rather than shelling out to cuobjdump. Only a
            # completely absent .sass falls back to on-demand disassembly.
            shipped = B / suite / f"{cub.name}.sass"
            if shipped.exists():
                sass = shipped if shipped.stat().st_size > _MIN_SASS else None
            else:
                sass = _dump_sass(cub, SASS_CACHE / suite / f"{cub.name}.sass")
            if sass:
                out.append((f"{suite}/{cub.name}", sass, cub))

    else:
        raise ValueError(f"unknown figure suite: {suite}")

    if kernel_filter:
        out = [k for k in out if kernel_filter in k[0]]
    return out
