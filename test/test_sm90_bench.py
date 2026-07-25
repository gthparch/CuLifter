"""
Test suite for SM90 benchmark kernels (cuBLAS, cuDNN, synthetic).

Discovers bench dirs under test/cublas_bench/, test/cudnn_bench/, and
test/sm90_synth/.  Each dir must contain a metadata.json.  The test
pipeline is:

  1. Locate SASS + ELF/cubin (pre-split in dir, or from mega-SASS)
  2. Lift to LLVM IR via main.py
  3. Compile IR + host_cpu.cpp with clang/clang++
  4. Run and check for "TEST PASSED" or "REF VERIFIED"

Benchmarks missing required files are skipped gracefully.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os
ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
SM = "90"

# Benchmark suites are discovered under bench/ (the downloaded library-suite
# store, SLIFTER_BENCH_ROOT to override) with a fallback to test/ for suites
# that still live in-tree. Each entry maps a discovered suite dir to its
# bench_id prefix; the framework itself stays in test/ and is arch-neutral.
# Default to test/ so `pytest` alone only touches the repo's tracked tests;
# developers can override SLIFTER_BENCH_ROOT to reach separately stored
# closed workloads (cuBLAS CB_ / cuDNN CD_).
BENCH_ROOT = Path(os.environ.get("SLIFTER_BENCH_ROOT", ROOT / "test"))
BENCH_SUITES = [
    # (bench_id prefix, [candidate dirs in priority order])
    ("cublas_bench", [BENCH_ROOT / "cublas", TEST_DIR / "cublas_bench"]),
    ("cudnn_bench",  [BENCH_ROOT / "cudnn",  TEST_DIR / "cudnn_bench"]),
    ("sm90_synth",   [BENCH_ROOT / "sm90_synth", TEST_DIR / "sm90_synth"]),
]

# ---------------------------------------------------------------------------
# Helpers (intentionally self-contained to avoid coupling to test_utils.py)
# ---------------------------------------------------------------------------

def _run_cmd(
    cmd: List[str],
    cwd: Path,
    timeout: int = 120,
    env: Optional[Dict] = None,
) -> subprocess.CompletedProcess:
    """Run a command, returning the CompletedProcess.  Raises on timeout."""
    try:
        r = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"Timeout ({timeout}s): {' '.join(map(str, cmd))}\n"
            f"stdout={exc.stdout}\nstderr={exc.stderr}"
        )
    return r


def _run_cmd_checked(
    cmd: List[str],
    cwd: Path,
    timeout: int = 120,
    env: Optional[Dict] = None,
) -> str:
    """Run a command, assert rc==0, return stdout."""
    r = _run_cmd(cmd, cwd, timeout, env=env)
    assert r.returncode == 0, (
        f"Failed (rc={r.returncode}): {' '.join(map(str, cmd))}\n"
        f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    return r.stdout


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_bench_dirs() -> List[Tuple[str, Path, dict]]:
    """Return list of (bench_id, bench_path, metadata) for all bench dirs."""
    found = []
    for suite_dir_name, candidates in BENCH_SUITES:
        suite_dir = next((c for c in candidates if c.exists()), None)
        if suite_dir is None:
            continue
        for bench_path in sorted(suite_dir.iterdir()):
            if not bench_path.is_dir():
                continue
            meta_path = bench_path / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            bench_id = f"{suite_dir_name}/{bench_path.name}"
            found.append((bench_id, bench_path, meta))
    return found


# Discover once at import time so parametrize has the list.
_ALL_BENCHES = _discover_bench_dirs()
_BENCH_IDS = [b[0] for b in _ALL_BENCHES]
_BENCH_MAP: Dict[str, Tuple[Path, dict]] = {
    b[0]: (b[1], b[2]) for b in _ALL_BENCHES
}


# ---------------------------------------------------------------------------
# SASS / ELF resolution
# ---------------------------------------------------------------------------

def _find_sass_elf(bench_path: Path, meta: dict) -> Optional[Tuple[Path, Optional[Path]]]:
    """Return (sass_path, elf_path-or-None) or None.

    Two modes:
      1. Pre-split: bench dir has kernel.sass + kernel.cubin (or .elf)
      2. Mega-SASS: metadata['sass_source'] points to a big SASS file,
         and a matching cubin exists at the parallel cubins path.
    """

    # -- Mode 1: pre-split files in the bench dir --------------------------
    sass_files = sorted(bench_path.glob("*.sass"))
    for sf in sass_files:
        # Prefer .elf companion, fall back to .cubin
        elf = sf.with_suffix(".elf")
        if elf.exists():
            return (sf, elf)
        cubin = sf.with_suffix(".cubin")
        if cubin.exists():
            return (sf, cubin)
    if sass_files:
        # No companion binary: single-file nvdisasm mode (closed-source
        # library splits — e.g. cuDNN — ship no per-kernel cubin).
        return (sass_files[0], None)

    # -- Mode 2: mega-SASS from benchmark data -----------------------------
    sass_source = meta.get("sass_source")
    if sass_source:
        sass_path = ROOT / sass_source
        if sass_path.exists():
            # Prefer explicit cubin path from metadata (per-kernel cubin)
            meta_cubin = meta.get("cubin")
            if meta_cubin:
                mc = Path(meta_cubin)
                if mc.exists():
                    return (sass_path, mc)
            # Fallback: derive cubin path by parallel-tree substitution
            cubin_path = Path(
                str(sass_path)
                .replace("/sass/", "/cubins/")
                .replace(".sass", ".cubin")
            )
            if cubin_path.exists():
                return (sass_path, cubin_path)
            # Also try .elf
            elf_path = Path(str(cubin_path).replace(".cubin", ".elf"))
            if elf_path.exists():
                return (sass_path, elf_path)
            # Mega-SASS without a sibling cubin: scan the per-version cubins
            # directory (e.g. .../cubins/cublas/sm_80/) for the one that contains
            # the requested kernel symbol.  Cache the hit via meta['cubin'].
            kernel_func = meta.get("kernel_func")
            if kernel_func:
                cubin_dir = cubin_path.parent
                if cubin_dir.exists():
                    # Strip trailing template hash; the printable prefix is enough.
                    needle = kernel_func[:60].encode()
                    for c in sorted(cubin_dir.glob("*.cubin")):
                        try:
                            if needle in c.read_bytes():
                                return (sass_path, c)
                        except OSError:
                            continue

    return None


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _lift_sass(sass_path: Path, elf_path: Optional[Path], out_dir: Path, bench_name: str, kernel_func: Optional[str] = None) -> Path:
    """Lift SASS(+ELF) to LLVM IR via main.py.  Returns path to .ll file."""
    out_base = out_dir / f"{bench_name}_x86_sm{SM}"
    out_ll = out_base.with_suffix(".ll")

    cmd = [
        "python", str(ROOT / "main.py"),
        "-i", str(sass_path),
        *([str(elf_path)] if elf_path is not None else []),
        "-o", str(out_base),
        "--lifter", "x86",
    ]
    if kernel_func:
        cmd += ["-name", kernel_func]
    _run_cmd_checked(cmd, cwd=ROOT, timeout=300)

    assert out_ll.exists(), f"Lifter did not produce {out_ll}"
    return out_ll


def _compile_x86(ll_path: Path, host_cpp: Path, out_dir: Path, bench_name: str) -> Path:
    """Compile .ll + host_cpu.cpp -> executable."""
    obj_path = out_dir / f"{bench_name}.o"
    exe_path = out_dir / f"{bench_name}.out"

    # Compile IR to object
    _run_cmd_checked(
        ["clang", "-O2", "-fPIC", "-target", "x86_64", "-c", str(ll_path), "-o", str(obj_path)],
        cwd=ROOT,
    )

    # Link with host driver
    wrapper_src = TEST_DIR / "common" / f"kernel_wrapper_{SM}.h"

    # Create a temp include dir with kernel_wrapper.h pointing to the SM90 version
    inc_dir = out_dir / "inc"
    inc_dir.mkdir(exist_ok=True)
    shutil.copy(wrapper_src, inc_dir / "kernel_wrapper.h")

    # g++ for the host driver: conda clang++ on some hosts does not resolve
    # libstdc++ headers (<cstdint> etc.) — same policy as test_utils.py.
    import os
    cxx = os.environ.get("SLIFTER_HOST_CXX", "g++")
    link_cmd = [
        cxx, "-std=c++20", "-O2",
        f"-D__CUDA_ARCH__={SM}0",
        f"-I{inc_dir}",
        f"-I{TEST_DIR / 'common'}",
        str(host_cpp),
        str(obj_path),
        "-o", str(exe_path),
        "-lpthread",
    ]
    _run_cmd_checked(link_cmd, cwd=ROOT)
    assert exe_path.exists(), f"Linker did not produce {exe_path}"
    return exe_path


def _run_exe(exe_path: Path, bench_path: Path, timeout: int = 60) -> Tuple[str, int]:
    """Run the compiled executable from the bench dir (so ref_output.bin is found)."""
    r = _run_cmd([str(exe_path)], cwd=bench_path, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


# ---------------------------------------------------------------------------
# Parametrized test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bench_id", _BENCH_IDS, ids=lambda b: b)
def test_bench_x86(bench_id, tmp_path):
    """End-to-end: lift -> compile -> run for one bench kernel."""
    bench_path, meta = _BENCH_MAP[bench_id]
    bench_name = meta.get("name", bench_path.name)

    # --- Check host driver ---------------------------------------------------
    host_cpp = bench_path / "host_cpu.cpp"
    if not host_cpp.exists():
        pytest.skip(f"No host_cpu.cpp in {bench_id}")

    # --- Locate SASS + ELF ---------------------------------------------------
    # If host_cpu.cpp advertises native_arch=sm80 but metadata points at the
    # sm_90 mega-SASS, switch to the sm_80 mega-SASS so param offsets (0x160 vs
    # 0x210) match what the host writes.
    host_text = host_cpp.read_text()
    if "Native arch: sm80" in host_text or "ARG_BASE = 0x160" in host_text:
        ss = meta.get("sass_source", "")
        if "sm_90" in ss:
            sm80_src = ss.replace("/sm_90/", "/sm_80/").replace("_sm90", "_sm80")
            if (ROOT / sm80_src).exists():
                meta = dict(meta)
                meta["sass_source"] = sm80_src
                meta.pop("cubin", None)  # invalidate sm90-specific cubin pin
    sass_elf = _find_sass_elf(bench_path, meta)
    if sass_elf is None:
        pytest.skip(f"No SASS+ELF available for {bench_id}")
    sass_path, elf_path = sass_elf

    # --- Stage 1: Lift --------------------------------------------------------
    # Collect all kernel symbols the host driver expects (it may launch multiple
    # kernels — e.g. dot_kernel + reduce_1Block_kernel for ddot/sdot).
    kernels = set()
    if meta.get("kernel_func"):
        kernels.add(meta["kernel_func"])
    import re as _re
    for m in _re.finditer(r'extern\s+"C"\s+void\s+(\w+)\s*\(', host_cpp.read_text()):
        kernels.add(m.group(1))
    kernel_arg = ",".join(sorted(kernels)) if kernels else None
    ll_path = _lift_sass(sass_path, elf_path, tmp_path, bench_name, kernel_arg)

    # --- Stage 2: Compile -----------------------------------------------------
    exe_path = _compile_x86(ll_path, host_cpp, tmp_path, bench_name)

    # --- Stage 3: Run and validate --------------------------------------------
    output, rc = _run_exe(exe_path, bench_path)

    # Accept either "TEST PASSED" (full lifted kernel) or "REF VERIFIED" (ref-only host)
    passed = ("TEST PASSED" in output) or ("REF VERIFIED" in output)
    assert passed, (
        f"Bench {bench_id} did not produce expected output.\n"
        f"rc={rc}\noutput:\n{output}"
    )


@pytest.mark.parametrize("bench_id", _BENCH_IDS, ids=lambda b: b)
def test_bench_lift_only(bench_id, tmp_path):
    """Lift-only: verify SASS -> LLVM IR succeeds (no host driver needed)."""
    bench_path, meta = _BENCH_MAP[bench_id]
    bench_name = meta.get("name", bench_path.name)

    # --- Locate SASS + ELF ---------------------------------------------------
    sass_elf = _find_sass_elf(bench_path, meta)
    if sass_elf is None:
        pytest.skip(f"No SASS+ELF available for {bench_id}")
    sass_path, elf_path = sass_elf

    # --- Lift only ------------------------------------------------------------
    ll_path = _lift_sass(sass_path, elf_path, tmp_path, bench_name, meta.get("kernel_func"))
    assert ll_path.exists() and ll_path.stat().st_size > 0, (
        f"Lift produced empty .ll for {bench_id}"
    )
