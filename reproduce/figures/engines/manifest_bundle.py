#!/usr/bin/env python3
"""Kernel discovery driven by the published corpus manifest.

`reproduce/sm90_manifest.json` describes the evaluated corpus: every
lift unit with its SASS/binary paths, SHA-256 digests, and FlashAttention
kernel selectors. This module enumerates those units from the installed
`benchmark/` bundle for the figure engines. Units whose shipped SASS is
absent or empty (PTX-only modules) are reported rather than silently
skipped.
"""
import json
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
BUNDLE = Path(os.environ.get("SLIFTER_BENCHMARK_DIR", _REPO / "benchmark"))
MANIFEST = Path(os.environ.get("SLIFTER_MANIFEST",
                               _REPO / "reproduce" / "sm90_manifest.json"))
_MIN_SASS = 200   # bytes; smaller means a failed/stub disassembly (bench_bundle rule)

# figure-suite key (FIG_SUITES) -> manifest suite key
SUITE_KEY = {
    "hecbench": "hecbench",
    "cuobjdump": "cuda_sdk",
    "cublas": "cublas",
    "cutlass": "cutlass",
    "flash_attn": "flash_attention",
    "cudnn_cnn": "cudnn_cnn",
    "cudnn_ops": "cudnn_ops",
    "cudnn_adv": "cudnn_adv",
}

# Paper Table 6 kernel counts per suite (validated by the manifest's
# totals.paper_kernels == 1681). Used by the exporter to attribute kernel
# counts to domains.
PAPER_KERNELS = {
    "hecbench": 229, "cublas": 972, "cutlass": 105, "flash_attn": 63,
    "cudnn_cnn": 7, "cudnn_ops": 256, "cudnn_adv": 30, "cuobjdump": 19,
}


def load_manifest():
    with open(MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


def _display_name(fig_suite, unit):
    # Mirrors bench_bundle naming so bench_domains.get_domain sees the same
    # strings: bare names for hecbench/cuda_sdk/flash dirs, suite-prefixed
    # names for the library and cutlass suites.
    name = unit["name"]
    if fig_suite in ("hecbench", "cuobjdump", "flash_attn"):
        return name
    return f"{fig_suite}/{name}"


def discover_nvlib_inputs():
    """98 NPP/nvJPEG kernels for the Figure 8b ablation:
    (name, kernel.sass, host_cpu.cpp driver, module cubin)."""
    root = BUNDLE / "nv_libraries"
    cubins = {g: root / "cubins" / f"{g}.cubin"
              for g in ("nvjpeg", "npp", "npp_compress", "npp_uf")}
    for cubin in cubins.values():
        if not cubin.is_file():
            raise FileNotFoundError(cubin)
    if not (root / "nvlib_compare.h").is_file():
        raise FileNotFoundError(root / "nvlib_compare.h")
    out = []
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        if directory.name == "cubins":
            continue
        sass = directory / "kernel.sass"
        driver = directory / "host_cpu.cpp"
        if not sass.is_file() or not driver.is_file():
            raise FileNotFoundError(f"incomplete nv_libraries input: {directory}")
        name = directory.name
        if name.startswith("npp_compress_"):
            group = "npp_compress"
        elif name.startswith("npp_uf_"):
            group = "npp_uf"
        elif name.startswith("npp_"):
            group = "npp"
        else:
            group = "nvjpeg"
        out.append((name, sass, driver, cubins[group]))
    return out


def discover_kernels(fig_suite: str, kernel_filter: str = "", output_dir=None):
    """Return (name, sass_path, elf_path) for every manifest unit of a suite."""
    manifest = load_manifest()
    mkey = SUITE_KEY[fig_suite]
    out, missing, empty = [], [], []
    for unit in manifest["units"]:
        if unit["suite"] != mkey:
            continue
        sass = BUNDLE / unit["sass"]
        binary = BUNDLE / unit["binary"] if unit.get("binary") else None
        # cutlass/cuda_sdk ship nvdisasm TEXT dumps: lifted as Asm (elf None),
        # exactly as bench_bundle does for those suites.
        elf = None if fig_suite in ("cutlass", "cuobjdump") else binary
        if not sass.is_file():
            missing.append(unit["name"])
            continue
        if sass.stat().st_size <= _MIN_SASS:
            empty.append(unit["name"])
            continue
        if elf is not None and not Path(elf).is_file():
            missing.append(unit["name"])
            continue
        out.append((_display_name(fig_suite, unit), sass, elf))
    if missing or empty:
        print(f"  [manifest] {fig_suite}: {len(missing)} unit(s) missing inputs "
              f"{missing[:5]}, {len(empty)} empty-SASS unit(s) {empty[:5]}")
    if kernel_filter:
        out = [k for k in out if kernel_filter in k[0]]
    return out


def coverage_report():
    """Units discovered vs. manifest, per suite — for the run log / AE."""
    manifest = load_manifest()
    rows = []
    for fig_suite, mkey in SUITE_KEY.items():
        want = sum(1 for u in manifest["units"] if u["suite"] == mkey)
        have = len(discover_kernels(fig_suite))
        rows.append((fig_suite, want, have, PAPER_KERNELS[fig_suite]))
    return rows


if __name__ == "__main__":
    total_u = total_have = total_k = 0
    print(f"{'suite':12s} {'units':>5s} {'found':>5s} {'paper_kernels':>13s}")
    for suite, want, have, kernels in coverage_report():
        print(f"{suite:12s} {want:5d} {have:5d} {kernels:13d}")
        total_u += want; total_have += have; total_k += kernels
    print(f"{'TOTAL':12s} {total_u:5d} {total_have:5d} {total_k:13d}")
