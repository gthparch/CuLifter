#!/usr/bin/env python3
"""Parallel hecbench end-to-end test runner for SM90 x86 lifting.

Runs each hecbench kernel through: lift → compile → link → run.
Supports parallel execution via -j flag.

Usage:
    python scripts/hecbench/run_hecbench.py -j 4          # all kernels, 4 workers
    python scripts/hecbench/run_hecbench.py --prefix relu  # only matching kernels
    python scripts/hecbench/run_hecbench.py --fails-only   # re-run only failures
"""

import argparse
import json
import os
import re
import signal
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BENCHMARK_DIR = Path(os.environ.get(
    "SLIFTER_BENCHMARK_DIR", PROJECT_ROOT / "benchmark"))
HECBENCH_DIR = (BENCHMARK_DIR if BENCHMARK_DIR.name == "hecbench"
                else BENCHMARK_DIR / "hecbench")
SASS_DIR = HECBENCH_DIR
CUBIN_DIR = HECBENCH_DIR / "cubins"
REF_IR_DIR = HECBENCH_DIR / "llvm_ir" / "sm_90"
COMMON_DIR = PROJECT_ROOT / "test" / "common"
TEMP_DIR = PROJECT_ROOT / "temp"
OUT_DIR = TEMP_DIR / "hec_run"
HINTS_FILE = Path(__file__).with_name("driver_hints.json")
DRIVER_DIR = Path(__file__).with_name("drivers")

SM = "90"


def load_driver_hints():
    """Load the tracked input/launch contracts.

    The downloaded benchmark bundle is data, not harness configuration.
    Keeping every override in one tracked map prevents a stale bundle-local
    driver or hint file from silently changing the verification performed.
    """
    try:
        all_hints = json.loads(HINTS_FILE.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load {HINTS_FILE}: {exc}") from exc
    if not isinstance(all_hints, dict):
        raise RuntimeError(f"invalid {HINTS_FILE}: expected an object")
    return all_hints


def get_driver_hints(name):
    """Return the tracked input/launch contract for one benchmark."""
    all_hints = load_driver_hints()
    hints = all_hints.get(name, {})
    if not isinstance(hints, dict):
        raise RuntimeError(f"invalid hints for {name}: expected an object")
    return hints


def run_cmd(cmd, timeout=60, cwd=None, env=None):
    proc = None
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=cwd, env=env, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        # Terminate the whole process group. Killing only the compiler driver
        # leaves cc1plus running and contaminates later parallel cases.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
        return -1, "", "timeout"
    except FileNotFoundError:
        return -1, "", f"executable not found: {cmd[0]}"


def summarize_error(stderr, fallback):
    """Return the actionable compiler/lifter line, not a generic trailer."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    for marker in (
        "fatal error:",
        "undefined reference",
        "executable not found:",
        "error:",
        "timeout",
    ):
        for line in lines:
            if marker in line.lower():
                return line[:500]
    return lines[-1][:500] if lines else fallback


def get_kernel_list():
    sass_names = {path.stem for path in SASS_DIR.glob("*.sass")}
    cubin_names = {path.stem for path in CUBIN_DIR.glob("*.cubin")}
    missing_cubin = sorted(sass_names - cubin_names)
    missing_sass = sorted(cubin_names - sass_names)
    if missing_cubin or missing_sass:
        details = []
        if missing_cubin:
            details.append("missing cubin: " + ", ".join(missing_cubin))
        if missing_sass:
            details.append("missing SASS: " + ", ".join(missing_sass))
        raise RuntimeError("; ".join(details))
    return sorted(sass_names)


def extract_kernel_sig(ref_ir_path):
    """Extract kernel function name from reference LLVM IR."""
    text = ref_ir_path.read_text()
    m = re.search(r'define\s+\w+\s+void\s+@"?(_Z\w+)"?\s*\(', text)
    if m:
        return m.group(1)
    m = re.search(r'define\s+\w+\s+void\s+@(\w+)\s*\(', text)
    if m:
        return m.group(1)
    return None


def get_lifted_functions(ll_path):
    """Extract function names from lifted LLVM IR (excludes _call_ suffixed
    clones and BPT.TRAP placeholder stubs)."""
    funcs = []
    try:
        text = ll_path.read_text()
    except Exception:
        return funcs
    for chunk in re.split(r'\ndefine\s+', '\n' + text)[1:]:
        m = re.match(r'void\s+@"?([^"(\s]+)"?\s*\(', chunk)
        if not m:
            continue
        name = m.group(1)
        # Skip internal call clones (e.g. _Z...kernel_call_3090)
        if '_call_' in name:
            continue
        # Skip arch placeholder stubs: cuDNN ships kernels whose whole body
        # is BPT.TRAP+spin (lifted: entry block calls llvm.trap). Launching
        # one is a guaranteed SIGILL, on GPU and CPU alike.
        body = chunk.split('{', 1)
        if len(body) == 2:
            entry = body[1].lstrip().split('\n')[:3]
            if any('llvm.trap' in ln for ln in entry):
                continue
        # CUTLASS arch-conditional guard stubs are the same disease with a
        # prologue: the kernel body is vprintf("ERROR : Arch conditional MMA
        # instruction used without targeting appropriate compute capability")
        # followed by BPT.TRAP (cubin built for sm_90; wgmma needs sm_90a).
        # llvm.trap then sits in the SECOND block, past the 3-line window
        # above. Recognize them structurally: every ret in the function is
        # immediately preceded by llvm.trap, i.e. no exit path completes.
        if len(body) == 2:
            fn_lines = [ln.strip()
                        for ln in body[1].split('\n}', 1)[0].split('\n')]
            ret_idx = [i for i, ln in enumerate(fn_lines)
                       if ln.startswith('ret ')]
            if ret_idx and all(
                    any('llvm.trap' in fn_lines[j]
                        for j in range(max(0, i - 2), i))
                    for i in ret_idx):
                continue
        funcs.append(name)
    return funcs


def _strip_static_prefix(name):
    """Remove _ZL prefix → _Z for matching static vs non-static names."""
    if name.startswith('_ZL'):
        return '_Z' + name[3:]
    return name


def _extract_param_suffix(mangled_name):
    """Extract the parameter encoding suffix from an Itanium-mangled name.

    For '_Z13my_kernel_fnPfS_i', returns 'PfS_i'.
    For '_ZL10static_funPKii', returns 'PKii'.
    """
    # Match _Z or _ZL, then <length><name>, rest is params
    m = re.match(r'_ZL?(\d+)', mangled_name)
    if not m:
        return mangled_name
    name_len = int(m.group(1))
    # Skip past: prefix + digits + name characters
    start = m.end() + name_len
    if start >= len(mangled_name):
        return ''
    return mangled_name[start:]


def _score_function_testability(mangled_name):
    """Score a function by how easy it is to test with an auto-generated driver.

    Higher = better (more testable). Penalizes struct params, bool, void params,
    and volatile pointers which the driver generator cannot handle well.

    Uses heuristics on the Itanium-mangled parameter suffix to detect complex types.
    """
    score = 100
    params_str = _extract_param_suffix(mangled_name)

    if not params_str:
        # No params at all — probably unmangled extern "C" name, neutral score
        return score

    # Detect struct-like params: digit(s) followed by a name (e.g., 6float4, 11int_fastdiv)
    # In Itanium mangling, struct/class types are encoded as <length><name>
    structs = re.findall(r'\d+[A-Za-z]', params_str)
    score -= len(structs) * 20

    # Penalize void params (empty kernel — mangled suffix 'v' means void)
    if params_str == 'v':
        score -= 50

    # Penalize bool params ('b' in Itanium = bool, but not inside struct names)
    # Remove struct name encodings first, then check for bare 'b'
    bare_params = re.sub(r'\d+\w+', '', params_str)
    if 'b' in bare_params:
        score -= 10

    # Penalize volatile pointers (PV = pointer-to-volatile)
    score -= params_str.count('PV') * 15

    # Bonus for simple pointer params (Pf = float*, Pi = int*, Pd = double*)
    simple_ptrs = len(re.findall(r'P[KV]*[fid]', params_str))
    score += simple_ptrs * 5

    return score


def find_best_match(lifted_funcs, ref_ir_path):
    """Find the best function to test from lifted IR, matching against reference IR.

    Strategy:
    1. Get all reference IR functions
    2. Find lifted functions that match reference functions (accounting for _ZL/_Z mismatch)
    3. Score all candidates by testability and pick the highest-scoring one (BUG-052)
    """
    from utils.kernel_sig import extract_from_ll

    def best_testable(functions):
        from utils.kernel_sig import signature_from_mangled
        ranked = []
        for index, name in enumerate(functions):
            sig = signature_from_mangled(name)
            has_pointer = bool(sig and any(p.is_pointer for p in sig.params))
            has_output = bool(sig and any(p.is_output for p in sig.params))
            has_params = bool(sig and sig.params)
            if has_output:
                half_params = sum(
                    "__half" in p.ctype or "bfloat" in p.ctype
                    for p in sig.params)
                ranked.append((
                    _score_function_testability(name) - 50 * half_params,
                    -len(sig.params), has_pointer, has_params, index, name))
        # Only override bundle order when doing so gives the driver an
        # observable output. A later scalar-only kernel adds no oracle and
        # may require more application global state than the first entry.
        return max(ranked)[-1] if ranked else (
            functions[0] if functions else None)

    if not ref_ir_path.exists():
        return best_testable(lifted_funcs)

    ref_sigs = extract_from_ll(str(ref_ir_path))
    ref_names = [s.name for s in ref_sigs]

    # Build lookup: stripped name → lifted name
    lifted_lookup = {}
    for lf in lifted_funcs:
        lifted_lookup[_strip_static_prefix(lf)] = lf

    # Collect all candidates: lifted functions that exist in reference IR
    candidates = []
    for rn in ref_names:
        stripped = _strip_static_prefix(rn)
        if stripped in lifted_lookup:
            lifted_name = lifted_lookup[stripped]
            score = _score_function_testability(rn)
            candidates.append((score, lifted_name))

    if candidates:
        # Pick the highest-scoring candidate (most testable)
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    return best_testable(lifted_funcs)


def _try_match_reference(kout, sig, init_hints=None):
    """Look for a reference.h next to the bench source and pick a
    matching function for `sig`. Returns (RefMatch, abs_path) or
    (None, None) when no usable reference is found.

    The bench's source dir is `HECBENCH_DIR / kout.name`; reference.h
    sits at the benchmark root alongside main.cu.
    """
    try:
        from utils.reference_match import parse_reference_h, match_reference
    except Exception:
        return None, None

    bench_name = kout.name
    ref_h = HECBENCH_DIR / bench_name / "reference.h"
    if not ref_h.exists():
        return None, None
    # The driver is compiled as plain C++ by g++: skip references that carry
    # CUDA-only constructs (__host__/__device__ qualifiers, __half math) —
    # they fail to compile when #included (crossEntropy's reference.h).
    try:
        htext = ref_h.read_text()
        if '__host__' in htext or '__device__' in htext or '__half' in htext:
            return None, None
        # Must be self-contained: some reference.h use macros that only
        # main.cu defines (ace's DATAYSIZE array dims) — including them
        # standalone fails to compile.
        defined = set(re.findall(r'#define\s+([A-Za-z_][A-Za-z0-9_]*)', htext))
        used_macroish = set(re.findall(r'\[\s*([A-Z_][A-Z0-9_]{2,})\s*\]', htext))
        if used_macroish - defined:
            return None, None
    except Exception:
        return None, None

    ref_funcs = parse_reference_h(ref_h)
    if not ref_funcs:
        return None, None

    # Read raw header source (comment-stripped) so the driver can embed
    # the entire reference.h. This carries over helper functions that
    # the matched function may call.
    try:
        from utils.reference_match import _strip_comments  # noqa: WPS437
        header_source = _strip_comments(ref_h.read_text(errors="replace"))
    except Exception:
        header_source = ""

    # Reconstruct the variable names that host_gen will assign to each
    # kernel parameter so we can pass them through to the matcher.
    # Also reconstruct the actual element types host_gen will allocate
    # for each pointer param (e.g. bool* -> int8_t*) so the matcher can
    # decide where to insert reinterpret_cast<> at the call site.
    var_names = []
    alloc_ctypes = []
    buf_idx = 0
    scalar_idx = 0
    int_count = 0
    for p in sig.params:
        if p.is_pointer:
            var_names.append(f"buf{buf_idx}")
            buf_idx += 1
            ct = p.ctype.replace('const ', '')
            if 'double' in ct:
                elem = 'double'
            elif 'long long' in ct or 'long' in ct:
                elem = 'int64_t'
            elif 'short' in ct:
                elem = 'int16_t'
            elif 'char' in ct:
                elem = 'int8_t'
            elif 'int' in ct:
                elem = 'int'
            elif 'bool' in ct:
                elem = 'int8_t'
            else:
                elem = 'float'
            alloc_ctypes.append(f"{elem}*")
        else:
            alloc_ctypes.append(p.ctype)
            if p.ctype == 'float':
                sh = ((init_hints or {}).get("scalars") or {}).get(
                    str(scalar_idx))
                scalar_idx += 1
                var_names.append(sh if sh else '2.0f')
            elif p.ctype == 'double':
                sh = ((init_hints or {}).get("scalars") or {}).get(
                    str(scalar_idx))
                scalar_idx += 1
                var_names.append(sh if sh else '2.0')
            elif p.ctype == 'bool':
                var_names.append('(bool)true')
            elif p.ctype in (
                    'int', 'int32_t', 'unsigned int', 'unsigned',
                    'uint32_t'):
                ih = ((init_hints or {}).get("int_scalars") or {}).get(
                    str(int_count))
                int_count += 1
                var_names.append(ih if ih else 'N')
            elif ('long' in p.ctype
                  or p.ctype in ('int64_t', 'uint64_t', 'size_t')):
                ih = ((init_hints or {}).get("int_scalars") or {}).get(
                    str(int_count))
                int_count += 1
                var_names.append(
                    f'static_cast<int64_t>({ih})'
                    if ih else 'static_cast<int64_t>(N)')
            else:
                var_names.append('N')

    rm = match_reference(sig, ref_funcs, var_names,
                        header_source=header_source,
                        alloc_ctypes=alloc_ctypes)
    if rm is None:
        return None, None
    return rm, str(ref_h)


def _detect_struct_ptr_slots(sig, kout):
    """IR-detected (ptr_slots, i64_scalar_slots, ptrptr_slots, idx_slots,
    count_slots) inside unknown-byval-struct param windows.

    Returns (None,) * 5 when the signature has no such params (nothing to
    scan), else the (possibly empty) slot lists from
    detect_param_ptr_slots over kout/kernel.ll. Requires apply_elf_arg_info
    to have pinned elf_offset and size — without ELF truth the windows are
    unknown and we don't guess.
    """
    from utils.kernel_sig import get_struct_fields
    windows = [(p.elf_offset, p.elf_offset + p.size) for p in sig.params
               if p.is_byval_struct and p.elf_offset >= 0 and p.size > 0
               and not get_struct_fields(p.struct_name)]
    if not windows:
        return None, None, None, None, None
    ll = Path(kout) / "kernel.ll"
    if not ll.exists():
        return None, None, None, None, None
    from utils.host_gen import detect_param_ptr_slots
    return detect_param_ptr_slots(ll, sig.name, windows)


def _detect_ctaid_y(kernel_name, kout):
    """True when the lifted kernel's body consumes SR_CTAID.Y.

    On the x86 path SRSubstitute rewrites `S2R Rx, SR_CTAID.Y` into a
    const-bank-0 read at 0x24 (the kernel_wrapper ABI slot), which lifts
    to a `getelementptr @const_mem, i64 0, i64 0, i64 36` in the
    function body — launch params own 0x0..0x38 and kernel args start at
    ARG_BASE, so a bank-0 GEP at 36 is definitively a ctaid.y read.
    Such kernels bake a 2D launch contract (bn_pointwise splits the
    batch over ctaid.y); generate_host pins gridDimY for them.

    None when kernel.ll is missing, oversized (whole-module cudnn_adv IR
    is >150MB — don't read it per kernel), or the function isn't found:
    not scanned, keep the 1D launch.
    """
    ll = Path(kout) / "kernel.ll"
    try:
        if not ll.exists() or ll.stat().st_size > (64 << 20):
            return None
        text = ll.read_text()
    except Exception:
        return None
    m = re.search(r'^define[^\n]*@"?%s"?\(' % re.escape(kernel_name), text,
                  re.MULTILINE)
    if not m:
        return None
    end = text.find('\n}', m.start())
    body = text[m.start():end if end != -1 else len(text)]
    return '@"const_mem", i64 0, i64 0, i64 36' in body


def generate_driver(lifted_funcs, ref_ir_path, kout, sm_mode="normal",
                    c4_offsets=None, elf_path=None,
                    allow_reference=True):
    """Generate a driver from the current lift, ELF metadata, and input hints.

    The normal HeCBench path always calls this with a fresh output directory;
    no generated driver from an earlier run is accepted as an input.
    """
    from utils.kernel_sig import extract_from_ll, apply_elf_arg_info
    from utils.host_gen import generate_host

    init_hints = get_driver_hints(kout.name)

    forced_func = init_hints.get("kernel")
    if forced_func:
        target_func = next(
            (name for name in lifted_funcs
             if name == forced_func
             or _strip_static_prefix(name) == _strip_static_prefix(forced_func)),
            None)
    else:
        target_func = find_best_match(lifted_funcs, ref_ir_path)
    if target_func is None:
        return None

    # Get signature from reference IR if possible
    if ref_ir_path.exists():
        ref_sigs = extract_from_ll(str(ref_ir_path))
        target_stripped = _strip_static_prefix(target_func)
        for sig in ref_sigs:
            if _strip_static_prefix(sig.name) == target_stripped:
                # Use reference signature but with the lifted function name
                sig.name = target_func
                apply_elf_arg_info(sig, elf_path)
                # Strict-comparison upgrade: if the bench ships a
                # reference.h with a CPU implementation we can match,
                # have the driver call it and compare element-wise.
                # Otherwise host_gen falls back to the smoke check.
                ref_match, ref_header = (
                    _try_match_reference(kout, sig, init_hints)
                    if allow_reference else (None, None))
                pslots, i64slots, ppslots, idxslots, cntslots = \
                    _detect_struct_ptr_slots(sig, kout)
                host_code = generate_host(sig, sm=int(SM), sm_mode=sm_mode,
                                          c4_offsets=c4_offsets,
                                          ref_match=ref_match,
                                          ref_header=ref_header,
                                          init_hints=init_hints,
                                          extra_ptr_slots=pslots,
                                          extra_i64_slots=i64slots,
                                          extra_ptrptr_slots=ppslots,
                                          extra_idx_slots=idxslots,
                                          extra_count_slots=cntslots,
                                          ctaid_y=_detect_ctaid_y(sig.name,
                                                                  kout))
                new_driver = kout / "driver.cpp"
                new_driver.write_text(host_code)
                return new_driver

    # Fallback A: no reference IR — the lifted symbol IS the mangled CUDA
    # name and encodes the full parameter list; build the signature from it.
    # (The benchmark bundle ships no llvm_ir/, so this is the primary path.)
    from utils.kernel_sig import signature_from_mangled
    sig = signature_from_mangled(target_func)
    if sig is not None:
        apply_elf_arg_info(sig, elf_path)
        ref_match, ref_header = (
            _try_match_reference(kout, sig, init_hints)
            if allow_reference else (None, None))
        pslots, i64slots, ppslots, idxslots, cntslots = \
            _detect_struct_ptr_slots(sig, kout)
        host_code = generate_host(sig, sm=int(SM), sm_mode=sm_mode,
                                  c4_offsets=c4_offsets,
                                  ref_match=ref_match,
                                  ref_header=ref_header,
                                  init_hints=init_hints,
                                  extra_ptr_slots=pslots,
                                  extra_i64_slots=i64slots,
                                  extra_ptrptr_slots=ppslots,
                                  extra_idx_slots=idxslots,
                                  extra_count_slots=cntslots,
                                  ctaid_y=_detect_ctaid_y(sig.name, kout))
        new_driver = kout / "driver.cpp"
        new_driver.write_text(host_code)
        return new_driver

    # Fallback B: generate minimal driver with no args
    host_code = f'''#include <cstdint>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void {target_func}();

int main() {{
    launchKernel({target_func}, 1, 256);
    std::cout << "TEST PASSED" << std::endl;
    return 0;
}}
'''
    new_driver = kout / "driver.cpp"
    new_driver.write_text(host_code)
    return new_driver


def _compile_and_link_driver(driver_src, obj_path, exe_path, kout):
    """Compile driver and link with kernel object. Returns (rc, detail)."""
    with tempfile.TemporaryDirectory(dir=str(TEMP_DIR)) as tmpd:
        wrapper = COMMON_DIR / f"kernel_wrapper_{SM}.h"
        if wrapper.exists():
            shutil.copy(wrapper, Path(tmpd) / "kernel_wrapper.h")

        # Device printf runtime: lifted device printf is a CALL to
        # `vprintf(fmt, packed_args)` — NOT libc's va_list ABI. Without
        # this object the import silently binds to libc's vprintf and
        # reads the packed buffer as a va_list (CUTLASS 02_dump_reg_shmem
        # SIGSEGV). Plain C compiled separately (kernel_wrapper_75.h
        # documents why it can't live in a C++ TU); our strong definition
        # takes precedence over libc's at link time, and kernels that
        # never printf simply don't call it.
        prt_obj = Path(tmpd) / "printf_runtime.o"
        cc = os.environ.get("HECBENCH_CC") or "gcc"
        rc, _, err = run_cmd([cc, "-O2", "-c",
                              str(COMMON_DIR / "printf_runtime.c"),
                              "-o", str(prt_obj)])
        if rc != 0:
            return rc, summarize_error(
                err, "printf_runtime compile failed")

        bench_src_dir = HECBENCH_DIR / kout.name
        cmd = [
            os.environ.get("HECBENCH_CXX") or "g++", "-std=c++20", "-O2",
            f"-D__CUDA_ARCH__={SM}0",
            f"-I{tmpd}", f"-I{COMMON_DIR}",
        ]
        # When the strict-validation path is in use the driver does
        # `#include "reference.h"`. Add the bench source dir to the
        # include search path so that resolves; benign no-op otherwise.
        if bench_src_dir.exists():
            cmd.append(f"-I{bench_src_dir}")
        cmd += [
            str(driver_src), str(obj_path), str(prt_obj),
            "-o", str(exe_path), "-lpthread",
        ]
        compile_timeout = int(os.environ.get(
            "HECBENCH_COMPILE_TIMEOUT", "300"))
        rc, _, err = run_cmd(cmd, timeout=compile_timeout)

    if rc != 0:
        return rc, summarize_error(err, "link failed")
    return 0, ""


def _run_exe(exe_path, timeout=None, env=None):
    """Run the test executable. Returns (status, detail)."""
    if timeout is None:
        # Sequential x86 emulation of large grids can legitimately need
        # minutes (CB_sasum precedent); 30s misclassifies those as hangs.
        timeout = int(os.environ.get("SLIFTER_RUN_TIMEOUT", "30"))
    if not exe_path.exists():
        return "link_fail", "exe not produced despite rc=0"
    rc, out, err = run_cmd([str(exe_path)], timeout=timeout, env=env)
    if rc == -1:
        if "not found" in err:
            return "link_fail", err
        return "timeout", "execution timed out"
    if rc > 128:
        sig = rc - 128
        signames = {11: "SIGSEGV", 6: "SIGABRT", 8: "SIGFPE"}
        return "crash", f"signal {signames.get(sig, sig)}"
    if rc < 0 and rc != -1:
        # Python subprocess returns -signal_number for signal-killed processes
        sig = -rc
        signames = {11: "SIGSEGV", 6: "SIGABRT", 8: "SIGFPE"}
        return "crash", f"signal {signames.get(sig, sig)}"
    if "SKIP_SHMEM" in out:
        # Wrapper shared-memory guard: the launch config's shared footprint
        # exceeds the sm_90 carveout — invalid on real hardware too (the CUDA
        # driver would refuse the launch). Checked before TEST PASSED: the
        # run is invalid regardless of what was printed earlier.
        first = [l for l in out.split('\n') if "SKIP_SHMEM" in l]
        return "skip_shmem", first[0][:160] if first else ""
    if "TEST PASSED" in out:
        return "pass", ""
    if "FAIL" in out:
        first_fail = [l for l in out.split('\n') if "FAIL" in l]
        return "wrong_result", first_fail[0] if first_fail else f"rc={rc}"
    if rc != 0:
        return "run_fail", f"rc={rc}: {out.strip()[:120]}"
    return "other", f"no TEST PASSED: {out.strip()[:120]}"


def _validation_mode(driver_path, hand_written=False):
    if hand_written:
        return "native_oracle"
    try:
        text = Path(driver_path).read_text()
    except Exception:
        return "unknown"
    if "Strict validation: compare lifted output" in text:
        return "native_oracle"
    if "driver_hints buf_init census" in text:
        return "contract_oracle"
    if "Validation: check output is not all zeros" in text:
        return "output_smoke"
    return "execution_only"


def test_one_kernel(name, out_dir_str, sm_mode="normal", retry_small=True):
    """Full pipeline for one hecbench kernel. Returns (name, status, detail).

    Args:
        sm_mode: "normal" or "small" — controls buffer/grid sizing in generated drivers.
        retry_small: If True and execution times out with normal mode, retry with
            a regenerated "small" driver (BUG-041).
    """
    out_dir = Path(out_dir_str)
    kout = out_dir / name
    kout.mkdir(parents=True, exist_ok=True)

    sass_file = SASS_DIR / f"{name}.sass"
    elf_file = CUBIN_DIR / f"{name}.cubin"
    ref_ir = REF_IR_DIR / f"{name}.ll"
    test_dir = HECBENCH_DIR / name

    if not sass_file.exists() or not elf_file.exists():
        return name, "no_sass", "missing SASS or ELF"

    # Per-test lockstep opt-in (tracked suite hints {"lockstep": true}): the
    # kernel relies on pre-Volta warp-synchronous shared memory (no
    # __syncwarp), so lift wraps shared accesses in warp_turn calls and exec
    # arms the runtime turnstile (SIMT round ordering per warp).
    lockstep = bool(get_driver_hints(name).get("lockstep"))
    lift_env = {**os.environ, "CULIFTER_LOCKSTEP": "1"} if lockstep else None
    exe_env = {**os.environ, "SLIFTER_LOCKSTEP": "1"} if lockstep else None

    # Step 1: Lift
    ll_path = kout / "kernel.ll"
    rc, out, err = run_cmd([
        sys.executable, "main.py",
        "-i", str(sass_file), str(elf_file),
        "-o", str(ll_path.with_suffix("")),
        "--lifter", "x86"
    ], timeout=int(os.environ.get("SLIFTER_LIFT_TIMEOUT", "300")),
       cwd=str(PROJECT_ROOT), env=lift_env)

    if rc != 0 or not ll_path.exists():
        detail = summarize_error(err, "unknown")
        if "timeout" in detail.lower() or rc == -1:
            return name, "lift_timeout", "lift timed out"
        return name, "lift_fail", detail

    # Get functions available in lifted IR
    lifted_funcs = get_lifted_functions(ll_path)
    if not lifted_funcs:
        if "Function :" not in sass_file.read_text(errors="replace"):
            return name, "pass", ""
        return name, "lift_fail", "no kernel functions in lifted IR"

    # BUG-039: Detect c[4] device variable accesses in lifted IR
    from utils.host_gen import detect_c4_offsets
    c4_offsets = detect_c4_offsets(ll_path)

    # Step 2: Compile
    obj_path = kout / "kernel.o"
    clang = os.environ.get("HECBENCH_CLANG") or "clang"
    compile_timeout = int(os.environ.get(
        "HECBENCH_COMPILE_TIMEOUT", "300"))
    rc, _, err = run_cmd([
        clang, "-O2", "-fPIC", "-target", "x86_64",
        "-c", str(ll_path), "-o", str(obj_path)
    ], timeout=compile_timeout)
    if rc != 0:
        detail = summarize_error(err, "compile failed")
        return name, "compile_fail", detail

    # Step 3: select a tracked algorithm-specific oracle or generate a fresh
    # driver. Drivers from the downloaded data bundle are never executed.
    host_cpp = DRIVER_DIR / f"{name}.cpp"
    hand_written = bool(host_cpp and host_cpp.exists())

    if hand_written:
        driver_src = host_cpp
    else:
        driver_src = generate_driver(
            lifted_funcs, ref_ir, kout,
            sm_mode=sm_mode, c4_offsets=c4_offsets,
            elf_path=elf_file)
        if not driver_src or not driver_src.exists():
            return name, "no_driver", "driver generation failed"

    # Copy final driver to output dir
    if driver_src != kout / "driver.cpp":
        shutil.copy(driver_src, kout / "driver.cpp")
    driver_src = kout / "driver.cpp"

    # Step 3b: Compile and link driver
    exe_path = kout / "test.out"
    link_rc, link_detail = _compile_and_link_driver(driver_src, obj_path, exe_path, kout)
    if link_rc != 0 and not hand_written \
            and _validation_mode(driver_src) == "native_oracle":
        # Some reference headers depend on declarations or macros supplied
        # only by the benchmark's CUDA host program. If the embedded native
        # oracle is not standalone C++, regenerate the same current driver
        # with the documented smoke oracle rather than failing compilation.
        driver_src = generate_driver(
            lifted_funcs, ref_ir, kout, sm_mode=sm_mode,
            c4_offsets=c4_offsets, elf_path=elf_file,
            allow_reference=False)
        if driver_src and driver_src.exists():
            link_rc, link_detail = _compile_and_link_driver(
                driver_src, obj_path, exe_path, kout)
    if link_rc != 0:
        return name, "link_fail", link_detail

    validation = _validation_mode(driver_src, hand_written=hand_written)

    # Step 4: Run (BUG-057: increased timeout from 30s to 60s for normal mode).
    # An explicit SLIFTER_RUN_TIMEOUT overrides both modes — generous-budget
    # retests were silently capped at 60s before this honored the env knob.
    env_rt = os.environ.get("SLIFTER_RUN_TIMEOUT")
    exe_timeout = int(env_rt) if env_rt else (300 if sm_mode == "normal" else 30)
    status, detail = _run_exe(exe_path, timeout=exe_timeout, env=exe_env)
    if status == "pass":
        detail = f"validation={validation}"

    # BUG-045: If execution timed out and we haven't already tried small mode,
    # retry with a regenerated small driver (reduced N/BUF_SIZE). Regeneration
    # no longer needs reference IR (mangled-signature Fallback A).
    # Hand-written drivers opt out: regenerating would swap in a vacuous
    # auto-driver whose non-timeout FAIL would mask the real timeout.
    if status == "timeout" and retry_small and sm_mode != "small" \
            and not hand_written:
        driver_src = generate_driver(
            lifted_funcs, ref_ir, kout, sm_mode="small",
            c4_offsets=c4_offsets, elf_path=elf_file)
        if driver_src and driver_src.exists():
            # Recompile with the small driver
            exe_small = kout / "test_small.out"
            link_rc, link_detail = _compile_and_link_driver(driver_src, obj_path, exe_small, kout)
            if link_rc == 0:
                status2, detail2 = _run_exe(
                    exe_small, timeout=int(env_rt) if env_rt else 30, env=exe_env)
                if status2 != "timeout":
                    # Small mode resolved the timeout — use its result
                    if status2 == "pass":
                        validation2 = _validation_mode(driver_src)
                        detail2 = f"validation={validation2}; small"
                    elif detail2:
                        detail2 += " (small)"
                    else:
                        detail2 = "(small)"
                    return name, status2, detail2

    return name, status, detail


def _worker(args):
    name, out_dir, sm_mode = args
    return test_one_kernel(name, out_dir, sm_mode=sm_mode)


def main():
    parser = argparse.ArgumentParser(description="Parallel hecbench SM90 x86 runner")
    parser.add_argument("-j", "--parallel", type=int, default=1)
    parser.add_argument("--prefix", type=str, default="",
                        help="Filter kernels by prefix (e.g. relu,softmax)")
    parser.add_argument("--fails-only", action="store_true",
                        help="Re-run only non-passing from previous run")
    parser.add_argument("--small", action="store_true",
                        help="Force small buffer mode (N=1, BUF_SIZE=64, blockDim=1) for all "
                             "kernels. Useful for timeout-prone kernels on x86.")
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--save-ll", type=str, default="",
                        help="Save lifted .ll files to this directory (e.g. results/lifted_ir/). "
                             "Also saves driver.cpp and a results summary CSV.")
    args = parser.parse_args()

    sm_mode = "small" if args.small else "normal"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        all_kernels = get_kernel_list()
        configured = set(load_driver_hints())
    except RuntimeError as error:
        parser.error(str(error))

    if not all_kernels:
        parser.error(f"no HeCBench inputs found under {SASS_DIR}")
    unknown_hints = sorted(configured - set(all_kernels))
    if unknown_hints:
        parser.error("driver hints refer to inputs absent from the inventory: "
                     + ", ".join(unknown_hints))
    unknown_drivers = sorted(
        path.stem for path in DRIVER_DIR.glob("*.cpp")
        if path.stem not in all_kernels)
    if unknown_drivers:
        parser.error("tracked drivers refer to inputs absent from the inventory: "
                     + ", ".join(unknown_drivers))
    if not args.prefix and not args.fails_only and len(all_kernels) != 229:
        parser.error(
            f"full HeCBench run requires 229 inputs, found {len(all_kernels)}")

    if args.prefix:
        prefixes = [p.strip() for p in args.prefix.split(",")]
        test_list = [k for k in all_kernels if any(k.startswith(p) for p in prefixes)]
    elif args.fails_only:
        test_list = []
        for name in all_kernels:
            exe = out_dir / name / "test.out"
            if not exe.exists():
                test_list.append(name)
                continue
            try:
                rc, out, _ = run_cmd([str(exe)], timeout=10)
                if rc != 0 or "TEST PASSED" not in out:
                    test_list.append(name)
            except Exception:
                test_list.append(name)
    else:
        test_list = all_kernels

    if not test_list:
        parser.error("the selected HeCBench input set is empty")

    print(f"\n{'='*60}")
    print(f"  hecbench runner: {len(test_list)} kernels"
          f"{f' (j={args.parallel})' if args.parallel > 1 else ''}")
    print(f"{'='*60}\n")

    results = {}
    status_counts = {}

    if args.parallel > 1:
        import multiprocessing
        work = [(name, str(out_dir), sm_mode) for name in test_list]
        with multiprocessing.Pool(args.parallel) as pool:
            for i, (name, status, detail) in enumerate(pool.imap_unordered(_worker, work), 1):
                results[name] = (status, detail)
                status_counts[status] = status_counts.get(status, 0) + 1
                tag = "PASS" if status == "pass" else f"{status.upper()}: {detail[:80]}"
                print(f"[{i:3d}/{len(test_list)}] {tag}: {name}", flush=True)
    else:
        for i, name in enumerate(test_list, 1):
            print(f"[{i:3d}/{len(test_list)}] {name:<40s}", end="", flush=True)
            _, status, detail = test_one_kernel(name, str(out_dir), sm_mode=sm_mode)
            results[name] = (status, detail)
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "pass":
                print(" PASS")
            else:
                print(f" {status.upper()}: {detail[:80]}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Results: {len(test_list)} kernels")
    for status in ["pass", "lift_fail", "lift_timeout", "compile_fail",
                    "link_fail", "crash", "wrong_result", "run_fail",
                    "timeout", "no_sig", "no_sass", "no_driver", "other"]:
        if status in status_counts:
            print(f"  {status:<15s}: {status_counts[status]}")
    total = sum(status_counts.values())
    passed = status_counts.get("pass", 0)
    print(f"\n  {passed}/{total} passed end-to-end")
    validation_counts = {}
    for status, detail in results.values():
        if status != "pass" or not detail.startswith("validation="):
            continue
        validation = detail.split("=", 1)[1].split(";", 1)[0]
        validation_counts[validation] = validation_counts.get(validation, 0) + 1
    if validation_counts:
        print("  validation:")
        for validation, count in sorted(validation_counts.items()):
            print(f"    {validation:<16s}: {count}")

    # Write results file
    results_file = out_dir / "results.txt"
    with open(results_file, "w") as f:
        for name in sorted(results):
            status, detail = results[name]
            f.write(f"{name}\t{status}\t{detail}\n")
    print(f"\nResults written to {results_file}")

    # Save lifted IR, drivers, and summary for paper/analysis
    if args.save_ll:
        save_dir = Path(args.save_ll)
        save_dir.mkdir(parents=True, exist_ok=True)
        ll_dir = save_dir / "lifted_ir"
        drv_dir = save_dir / "drivers"
        ll_dir.mkdir(exist_ok=True)
        drv_dir.mkdir(exist_ok=True)

        saved = 0
        for name in sorted(results):
            ll_src = out_dir / name / "kernel.ll"
            drv_src = out_dir / name / "driver.cpp"
            if ll_src.exists():
                shutil.copy(ll_src, ll_dir / f"{name}.ll")
                saved += 1
            if drv_src.exists():
                shutil.copy(drv_src, drv_dir / f"{name}_driver.cpp")

        # Write CSV summary
        csv_path = save_dir / "results.csv"
        with open(csv_path, "w") as f:
            f.write("kernel,status,detail,sass_lines,lifted_ir_lines,functions\n")
            for name in sorted(results):
                status, detail = results[name]
                # Count SASS lines
                sass_file = SASS_DIR / f"{name}.sass"
                sass_lines = 0
                if sass_file.exists():
                    try:
                        sass_lines = sum(1 for _ in open(sass_file))
                    except Exception:
                        pass
                # Count lifted IR lines and functions
                ll_file = ll_dir / f"{name}.ll"
                ir_lines = 0
                ir_funcs = 0
                if ll_file.exists():
                    try:
                        text = ll_file.read_text()
                        ir_lines = text.count('\n')
                        ir_funcs = text.count('define void @')
                    except Exception:
                        pass
                detail_csv = detail.replace(',', ';').replace('"', "'")[:100]
                f.write(f"{name},{status},{detail_csv},{sass_lines},{ir_lines},{ir_funcs}\n")

        print(f"\nSaved {saved} lifted IR files to {ll_dir}")
        print(f"Saved drivers to {drv_dir}")
        print(f"Summary CSV: {csv_path}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
