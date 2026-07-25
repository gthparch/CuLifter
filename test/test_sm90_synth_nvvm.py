"""
GPU-execution test suite for the test/sm90_synth microbench corpus (NVVM).

Every sm90_synth/<bench>/ dir with a standalone self-checking .cu (one that
defines main()) becomes one parametrized test:

  1. build the vanilla binary with `nvcc -arch=sm_90a -cudart=shared`
     (-cudart=shared is REQUIRED — static cudart embeds cudaLaunchKernel and
     defeats the LD_PRELOAD ptxswap interposition);
  2. dump SASS+ELF via cuobjdump from that same binary, so the lifted code
     corresponds 1:1 to what actually executes;
  3. lift with main.py --lifter nvvm;
  4. compile the lifted IR to PTX with llc -mcpu=sm_90a;
  5. run vanilla, then run again with the ptxswap shim preloaded
     (PTXSWAP_VERBOSE=1);
  6. assert swap count >= 1 (a zero-swap run FAILS — it would ride the
     vanilla cubin and pass vacuously), nondet-filtered stdout matches, and
     exit codes match.

sm_90a is used uniformly for build, lift target detection, and llc: the
wgmma/tma/stsm/mbarrier benches require the arch-specific ISA, and it is
harmless for the rest since the verification is differential.

Dirs without a main()-bearing .cu (kernel-only fragments consumed by the
combined build_*.sh drivers) are skipped. All build artifacts go to pytest's
tmp_path — the repo tree is never written, and stale outputs from retired
ad-hoc runs in benchmark directories are ignored entirely.
"""
import re
from pathlib import Path

import pytest

from .test_utils import (
    TEST_DIR,
    _dump_elf_from_exe,
    _lift_to_llvm,
    _run_synth_ptxswap,
    run_cmd,
)

# sm_90a uniformly (wgmma/tma/stsm need it; harmless for the others).
SM = "90a"
SYNTH_DIR = TEST_DIR / "sm90_synth"

_MAIN_RE = re.compile(r"\bint\s+main\s*\(")


def _standalone_cu(bench_dir: Path):
    """The standalone self-checking .cu of a bench dir: a .cu defining main().
    Prefers <dirname>.cu when several qualify (e.g. wgmma_min also carries
    probe/debug drivers); returns None for kernel-only dirs."""
    candidates = []
    for cu in sorted(bench_dir.glob("*.cu")):
        try:
            text = cu.read_text(errors="replace")
        except OSError:
            continue
        if _MAIN_RE.search(text):
            candidates.append(cu)
    if not candidates:
        return None
    for cu in candidates:
        if cu.stem == bench_dir.name:
            return cu
    return candidates[0]


def _discover_benches():
    if not SYNTH_DIR.exists():
        return []
    return sorted(d.name for d in SYNTH_DIR.iterdir() if d.is_dir())


@pytest.mark.parametrize("bench", _discover_benches())
def test_sm90_synth_nvvm(bench, tmp_path):
    bench_dir = SYNTH_DIR / bench
    cu = _standalone_cu(bench_dir)
    if cu is None:
        pytest.skip(f"No standalone .cu with main() in sm90_synth/{bench}")

    out_base = tmp_path / f"{cu.stem}_nvvm_sm{SM}"
    exe = tmp_path / f"{cu.stem}.bin"
    try:
        # (1) vanilla binary. -lcuda for the driver-API benches (TMA tensor
        # maps); -I<dir> for local headers next to the .cu.
        run_cmd([
            "nvcc", f"-arch=sm_{SM}", "-cudart=shared", "-lcuda",
            f"-I{bench_dir}",
            "-o", str(exe), str(cu),
        ], cwd=tmp_path, timeout=300)
        # (2) SASS+ELF from the very binary that will run.
        sass_path = out_base.with_suffix(".sass")
        sass_path.write_text(run_cmd(["cuobjdump", "--dump-sass", str(exe)], cwd=tmp_path))
        _dump_elf_from_exe(exe, out_base.with_suffix(".elf"), tmp_path)
    except (AssertionError, OSError) as e:
        # Missing/failing toolchain (no nvcc on login nodes, toolkit-version
        # rejects) is not a lifter signal — mirror the hecbench build policy.
        pytest.skip(f"Vanilla build failed for {bench} (toolchain issue, not a lifter signal): {str(e)[:300]}")

    # (3) lift — failures here are real lifter signals and must FAIL.
    out_ll = _lift_to_llvm(sass_path, "nvvm", out_base)

    # (4)-(6) llc -mcpu=sm_90a, vanilla + swapped runs, swap-count >= 1,
    # stdout compare (FP-tolerant), exit-code compare.
    _run_synth_ptxswap(exe, out_ll, SM, out_base.with_suffix(".ptx"),
                       out_base.with_suffix(".elf"), run_cwd=tmp_path, name=bench)
