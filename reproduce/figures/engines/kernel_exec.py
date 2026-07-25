#!/usr/bin/env python3
"""Shared lift + compile + link + run executor for the execution-based figures.

Both ablation runners (HeCBench and NPP/nvJPEG) use this pipeline. The
per-suite differences are the input inventory and the driver's include
directories. Drivers live with their inputs under
`benchmark/<suite>/<unit>/host_cpu.cpp`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = PROJECT_ROOT / "test" / "common"

STATUSES = ["pass", "wrong_result", "crash", "lift_fail",
            "compile_fail", "link_fail", "timeout"]

LIFT_TIMEOUT = int(os.environ.get("SLIFTER_EXEC_LIFT_TIMEOUT", "180"))
RUN_TIMEOUT = int(os.environ.get("SLIFTER_EXEC_RUN_TIMEOUT", "60"))


def _tail(text: str, limit: int = 300) -> str:
    return (text or "")[-limit:].replace("\n", " ")


def lift_kernel(sass_path, cubin_path, out_prefix, extra_env=None,
                timeout=None):
    """Lift one SASS input via main.py. Returns (ll_path | None, status, detail)."""
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"),
           "-i", str(sass_path), str(cubin_path),
           "-o", str(out_prefix), "--lifter", "x86"]
    env = {**os.environ, **(extra_env or {})}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout or LIFT_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return None, "timeout", f"lift timeout ({timeout or LIFT_TIMEOUT}s)"
    if proc.returncode:
        return None, "lift_fail", _tail(proc.stderr or proc.stdout)
    ll_path = Path(str(out_prefix) + ".ll")
    if not ll_path.is_file():
        return None, "lift_fail", "lifter produced no LLVM IR"
    return ll_path, "ok", ""


def _printf_runtime_obj(work_dir):
    """Compile the device-printf runtime shim (once per work dir), exactly as
    the HeCBench end-to-end lane does: it must take precedence over libc's
    printf at link time; kernels that never printf simply don't call it."""
    obj = Path(work_dir) / "printf_runtime.o"
    if obj.is_file():
        return obj, ""
    cc = os.environ.get("SLIFTER_CC", "gcc")
    proc = subprocess.run(
        [cc, "-O2", "-c", str(COMMON_DIR / "printf_runtime.c"),
         "-o", str(obj)],
        capture_output=True, text=True, timeout=120)
    if proc.returncode:
        return None, _tail(proc.stderr)
    return obj, ""


def execute_driver(ll_path, driver_path, include_dirs, work_dir,
                   run_timeout=None, exec_env=None):
    """Compile the lifted IR, link the unit's committed CPU driver, run it.
    Returns (status, detail) with status in STATUSES."""
    import time

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    obj = work_dir / "kernel.o"
    clang = os.environ.get("SLIFTER_CLANG", "clang")
    cmd = [clang, "-O2", "-fPIC", "-target", "x86_64",
           "-c", str(ll_path), "-o", str(obj)]
    for attempt in range(2):
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=120)
        if proc.returncode == 0:
            break
        # a shared-FS read race can surface a truncated .ll; retry once
        time.sleep(1.0)
    if proc.returncode:
        return "compile_fail", _tail(proc.stderr)

    prt_obj, err = _printf_runtime_obj(work_dir)
    if prt_obj is None:
        return "compile_fail", f"printf_runtime: {err}"

    include = work_dir / "include"
    include.mkdir(exist_ok=True)
    shutil.copy2(COMMON_DIR / "kernel_wrapper_90.h",
                 include / "kernel_wrapper.h")

    cxx = os.environ.get("SLIFTER_HOST_CXX")
    if not cxx:
        cxx = "/usr/bin/g++" if Path("/usr/bin/g++").is_file() else "g++"
    exe = work_dir / "run"
    cmd = [cxx, "-std=c++20", "-O2", "-D__CUDA_ARCH__=900",
           f"-I{include}"]
    cmd += [f"-I{d}" for d in include_dirs]
    cmd += [f"-I{COMMON_DIR}", str(driver_path), str(obj), str(prt_obj),
            "-o", str(exe), "-lpthread"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode:
        return "link_fail", _tail(proc.stderr)

    for attempt in range(4):
        try:
            proc = subprocess.run([str(exe)], capture_output=True, text=True,
                                  timeout=run_timeout or RUN_TIMEOUT,
                                  env={**os.environ, **(exec_env or {})})
            break
        except subprocess.TimeoutExpired:
            return "timeout", f"run timeout ({run_timeout or RUN_TIMEOUT}s)"
        except OSError as exc:
            # ETXTBSY: the freshly linked binary can still be busy on a
            # shared filesystem; back off and retry
            if exc.errno != 26 or attempt == 3:
                return "crash", f"exec failed: {exc}"
            time.sleep(0.5 * (attempt + 1))
    output = (proc.stdout or "") + (proc.stderr or "")
    if "TEST PASSED" in output:
        return "pass", ""
    if proc.returncode < 0:
        return "crash", f"signal {-proc.returncode}"
    if "FAIL" in output:
        return "wrong_result", _tail(output)
    return "crash", f"exit {proc.returncode}: {_tail(output)}"
