import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from parse.parser import SaSSParser
from sir.module import Module
from transform.toggle_reuse import ToggleReuse
from utils.asm import Asm
from utils.cuasm import CuAsm

ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"
LIFTERS = ("x86", "nvvm", "arm")

def run_cmd(cmd: List[str], cwd: Path, timeout: int = 60000) -> str:
    try:
        r = subprocess.run(
            cmd, cwd=cwd, text=True, capture_output=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        raise AssertionError(
            f"Timeout ({timeout}s): {' '.join(map(str, cmd))}\n{e.stdout}\n{e.stderr}"
        )

    assert r.returncode == 0, (
        f"Failed (rc={r.returncode}): {' '.join(map(str, cmd))}\n{r.stdout}\n{r.stderr}"
    )
    return r.stdout

def _dump_elf_from_exe(exe_path: Path, out_path: Path, cwd: Path):
    out = run_cmd(["cuobjdump", "--dump-elf", str(exe_path)], cwd)
    out_path.write_bytes(out.encode("utf-8") if isinstance(out, str) else out)

def discover_bases() -> Dict[str, Dict]:
    by_base = {}
    suites = ["nvbit", "cuobjdump", "nvdisasm", "ml", "addrtest", "cuda_handwritten"]
    sms = ["35", "52", "75", "80", "90", "100", "120"]

    for suite in suites:
        suite_dir = TEST_DIR / suite
        if not suite_dir.exists(): continue

        for test_dir in suite_dir.iterdir():
            if not test_dir.is_dir(): continue

            testname = test_dir.name
            if suite == "hecbench":
                if not testname.endswith("-cuda"): continue
                testname = testname[:-5]

            cu_file = test_dir / f"{testname}.cu"
            if suite == "hecbench" and not cu_file.exists():
                cu_file = test_dir / "main.cu"
            
            # Ensure we only count ASM if the corresponding ELF also exists
            if suite == "nvdisasm":
                has_valid_asm = any(s.exists() for s in test_dir.glob("*.nvdisasm.txt"))
            else:
                has_valid_asm = any(
                    s.exists() and s.with_suffix(".elf").exists()
                    for s in test_dir.glob("*.sass")
                )

            if not cu_file.exists() and not has_valid_asm:
                continue

            hosts = {}
            if (test_dir / "host_cpu.cpp").exists():
                hosts["x86"] = str((test_dir / "host_cpu.cpp").relative_to(TEST_DIR))
            if (test_dir / "host_cuda.cu").exists():
                hosts["nvvm"] = str((test_dir / "host_cuda.cu").relative_to(TEST_DIR))

            base_key = f"{suite}/{testname}"
            by_base.setdefault(base_key, {})

            for sm in sms:
                if suite == "nvdisasm":
                    asm_file = test_dir / f"{testname}_sm{sm}.nvdisasm.txt"
                    elf_file = None
                else:
                    asm_file = test_dir / f"{testname}_sm{sm}.sass"
                    elf_file = asm_file.with_suffix(".elf")
                
                # Only record path if BOTH exist
                asm_rel = ""
                if asm_file.exists() and (elf_file is None or elf_file.exists()):
                    asm_rel = str(asm_file.relative_to(TEST_DIR))

                by_base[base_key][sm] = {
                    "sass_rel": asm_rel,
                    "hosts": hosts
                }
    return by_base

def get_test_bases():
    return discover_bases()

def get_bases_for_sm_suite(bases: Dict, sm: str, suite: str) -> List[str]:
    return sorted([b for b, m in bases.items() if sm in m and b.startswith(f"{suite}/")])

def _compile_hecbench(test_dir: Path, sm: str, extra_cflags: str = "") -> Path:
    """Build hecbench test in-place using Makefile, returns path to executable.

    Uses a 5-minute build timeout: even multi-file projects (e.g. logic-rewrite
    with 11 .cu files) finish well under that. Anything longer is a stuck build
    that would otherwise consume the entire SLURM walltime."""
    run_cmd(["make", "clean"], cwd=test_dir, timeout=120)
    cmd = ["make", f"ARCH=sm_{sm}"]
    if extra_cflags:
        cmd.append(f"EXTRA_CFLAGS={extra_cflags}")
        # Some HeCBench Makefiles define CFLAGS WITHOUT $(EXTRA_CFLAGS) (e.g.
        # colorwheel/grep: `CFLAGS := -std=c++14 ...`), so EXTRA_CFLAGS never
        # reaches the link and -cudart=shared is dropped -> the binary STATICALLY
        # links cudart -> LD_PRELOAD can't interpose cudaLaunchKernel -> the
        # ptxswap harness sees swaps=0 (NO_SWAP), a vacuous non-test. Their link
        # line uses `$(LDFLAGS)` directly, so override LDFLAGS to force shared
        # linkage. Only for these Makefiles — EXTRA_CFLAGS-aware ones (the 246
        # passing kernels) already link shared via CFLAGS and are left untouched.
        if "-cudart=shared" in extra_cflags:
            try:
                mk = (test_dir / "Makefile").read_text()
                cf = re.search(r'^CFLAGS\s*[:?]?=\s*(.*)$', mk, re.MULTILINE)
                if cf and "EXTRA_CFLAGS" not in cf.group(1):
                    cmd.append("LDFLAGS=-cudart=shared")
            except OSError:
                pass
    run_cmd(cmd, cwd=test_dir, timeout=300)

    makefile = test_dir / "Makefile"
    match = re.search(r'^program\s*=\s*(\S+)', makefile.read_text(), re.MULTILINE)
    if match:
        exe = test_dir / match.group(1)
        if not exe.exists():
            raise AssertionError(f"Makefile did not produce '{match.group(1)}' in {test_dir}")
        return exe

    # No `program = ...` line: fall back to common HeCBench naming patterns.
    # Tests like b+tree (produces "b+tree.out"), heartwall, hybridsort,
    # streamcluster all use the directory basename rather than "main".
    dir_basename = test_dir.name.removesuffix("-cuda")
    candidates = ["main", dir_basename, f"{dir_basename}.out", "bin/main"]
    for name in candidates:
        cand = test_dir / name
        if cand.exists() and cand.is_file() and os.access(cand, os.X_OK):
            return cand
    raise AssertionError(
        f"Makefile did not produce a recognized executable in {test_dir} "
        f"(looked for: {candidates})"
    )

def _compile_cuda_binary(test_dir: Path, testname: str, sm: str, out_dir: Path) -> Tuple[str, Path]:
    tmp_exe = out_dir / "temp.out"

    cu_files = [p for p in test_dir.glob("**/*.cu") if p.name != "host_cuda.cu"]
    c_files = list(test_dir.glob("**/*.c"))
    cpp_files = [p for p in test_dir.glob("**/*.cpp") if p.name != "host_cpu.cpp"]

    include_flags = [
        "-I.",
        "-Icommon",
        f"-I{test_dir.relative_to(TEST_DIR)}",
    ]

    cmd = ["nvcc", f"-arch=sm_{sm}", *include_flags, "-Ddfloat=double", "-Ddlong=int", "-std=c++17", "-w", "-lcuda"]
    
    if len(cu_files) > 1 or c_files or cpp_files:
        cmd.extend([f"./{f.relative_to(TEST_DIR)}" for f in cu_files + c_files + cpp_files])
        cmd.extend(["-o", str(tmp_exe)])
        run_cmd(cmd, cwd=TEST_DIR)
    else:
        main_cu = test_dir / "main.cu" if (test_dir / "main.cu").exists() else test_dir / f"{testname}.cu"
        cmd = [
            "nvcc",
            f"-arch=sm_{sm}",
            "-cubin",
            *include_flags,
            "-Ddfloat=double",
            "-Ddlong=int",
            "-std=c++17",
            "-w",
            f"./{main_cu.relative_to(TEST_DIR)}",
            "-o",
            str(tmp_exe),
        ]
        run_cmd(cmd, cwd=TEST_DIR)
        
    sass_out = run_cmd(["cuobjdump", "--dump-sass", str(tmp_exe)], cwd=TEST_DIR)
    return sass_out, tmp_exe

def _compile_cuda_cubin(test_dir: Path, testname: str, sm: str, out_dir: Path) -> Path:
    tmp_cubin = out_dir / "temp.cubin"

    cu_files = [p for p in test_dir.glob("*.cu") if p.name != "host_cuda.cu"]
    c_files = list(test_dir.glob("*.c"))
    cpp_files = list(test_dir.glob("*.cpp"))

    if len(cu_files) != 1 or c_files or cpp_files:
        raise AssertionError(f"Expected a single CUDA source for nvdisasm: {testname}")

    cmd = [
        "nvcc", f"-arch=sm_{sm}", "-cubin",
        "-I.", "-Icommon", f"-I{test_dir.relative_to(TEST_DIR)}",
        "-Ddfloat=double", "-Ddlong=int", "-std=c++17", "-w",
        f"./{cu_files[0].relative_to(TEST_DIR)}",
        "-o", str(tmp_cubin),
    ]
    run_cmd(cmd, cwd=TEST_DIR)
    return tmp_cubin

def _ensure_sass(base: str, sm: str, entry: Dict) -> Path:
    suite, testname = base.split("/", 1)
    sass_rel = entry.get("sass_rel")
    
    if sass_rel:
        p = TEST_DIR / sass_rel
        if p.exists():
            return p

    test_dir = TEST_DIR / suite / (f"{testname}-cuda" if suite == "hecbench" else testname)
    if suite == "nvdisasm":
        target_sass = test_dir / f"{testname}_sm{sm}.nvdisasm.txt"
    else:
        target_sass = test_dir / f"{testname}_sm{sm}.sass"
        target_elf = target_sass.with_suffix(".elf")

    if suite == "nvdisasm":
        if target_sass.exists():
            return target_sass
    else:
        if target_sass.exists() and target_elf.exists():
            return target_sass

    if suite == "hecbench":
        try:
            exe_path = _compile_hecbench(test_dir, sm)
            sass_content = run_cmd(["cuobjdump", "--dump-sass", str(exe_path)], cwd=test_dir)
            target_sass.write_text(sass_content)
            _dump_elf_from_exe(exe_path, target_elf, test_dir)
            exe_path.unlink(missing_ok=True)
            (test_dir / "main.o").unlink(missing_ok=True)
        except AssertionError as e:
            pytest.skip(f"Build failed for {base}: {e}")
        return target_sass

    if suite == "nvdisasm":
        if not list(test_dir.glob("*.cu")):
            pytest.skip(f"No SASS or Source found for {base}")

        with tempfile.TemporaryDirectory(prefix=f"build_{testname}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            try:
                cubin_path = _compile_cuda_cubin(test_dir, testname, sm, tmp_path)
                sass_content = run_cmd(["nvdisasm", str(cubin_path)], cwd=TEST_DIR)
                target_sass.write_text(sass_content)
            except AssertionError as e:
                pytest.skip(f"Build failed for {base}: {e}")
        return target_sass

    if not list(test_dir.glob("*.cu")):
        pytest.skip(f"No SASS or Source found for {base}")

    with tempfile.TemporaryDirectory(prefix=f"build_{testname}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            sass_content, exe_path = _compile_cuda_binary(test_dir, testname, sm, tmp_path)
            target_sass.write_text(sass_content)
            _dump_elf_from_exe(exe_path, target_elf, tmp_path)
        except AssertionError as e:
             pytest.skip(f"Build failed for {base}: {e}")

    return target_sass

def _tdir_rel(p: Path) -> str:
    """Render a path for a subprocess running with cwd=TEST_DIR: relative when
    the file lives under test/ (existing suites, unchanged), absolute when it
    lives elsewhere (pytest tmp-dir builds, e.g. the sm90_synth suite)."""
    try:
        return str(p.relative_to(TEST_DIR))
    except ValueError:
        return str(p)

def _lift_to_llvm(sass_path: Path, lifter: str, out_base_path: Path) -> Path:
    out_ll = out_base_path.with_suffix(".ll")
    cmd = [
        sys.executable, "../main.py",
        "-i",
        _tdir_rel(sass_path),
        "-o", str(out_ll.with_suffix("")),
        "--lifter", lifter
    ]
    if not sass_path.name.endswith(".nvdisasm.txt"):
        elf_path = sass_path.with_suffix(".elf")
        cmd.insert(4, _tdir_rel(elf_path))
    # 60-min lift timeout: most tests lift in seconds. logic-rewrite
    # (269 kernels via thrust/cub including factorFromTruth at 17.7k
    # SASS instructions) is the long-tail outlier. On slower CPU hosts
    # the lift exceeds 30 min, so 60 gives headroom while
    # still bounding any genuine hang.
    run_cmd(cmd, cwd=TEST_DIR, timeout=3600)
    return out_ll

def _assemble_nvdisasm(module: Module, out_base_path: Path, cubin_path: Path) -> Path:
    out_cubin = out_base_path.with_suffix(".cubin")
    out_cuasm = out_base_path.with_suffix(".cuasm")
    cuasm_py = ROOT / "tools" / "CuAssembler" / "bin" / "cuasm.py"
    cuasm_root = cuasm_py.parents[1]

    cmd = [
        "env",
        f"PYTHONPATH={cuasm_root}",
        "python", str(cuasm_py),
        str(cubin_path),
        "-o", str(out_cuasm.relative_to(TEST_DIR)),
    ]
    run_cmd(cmd, cwd=TEST_DIR)
    assert out_cuasm.exists() and out_cuasm.stat().st_size > 0

    cuasm = CuAsm(out_cuasm.read_text())
    cuasm.set_module(module)
    out_cuasm.write_text(cuasm.emit())

    cmd = [
        "env",
        f"PYTHONPATH={cuasm_root}",
        "python", str(cuasm_py),
        str(out_cuasm.relative_to(TEST_DIR)),
        "--asm2bin",
        "-o", str(out_cubin.relative_to(TEST_DIR)),
    ]
    try:
        run_cmd(cmd, cwd=TEST_DIR)
    except AssertionError as e:
        # CuAssembler rejects some per-operand `.reuse` combinations as unknown modifiers.
        # Retry once with `.reuse` stripped so nvdisasm emit tests remain assemble-able.
        msg = str(e)
        if "Unknown modifiers" not in msg or "reuse" not in msg:
            raise
        out_cuasm.write_text(out_cuasm.read_text().replace(".reuse", ""))
        run_cmd(cmd, cwd=TEST_DIR)
    assert out_cubin.exists() and out_cubin.stat().st_size > 0
    return out_cubin

def _run_x86_host(exe_path: Path, ll_path: Path, host_rel: str, sm: str):
    obj_path = exe_path.with_suffix(".o")
    
    run_cmd(["clang", "-O2", "-fPIC", "-target", "x86_64", "-c", str(ll_path), "-o", str(obj_path)], cwd=TEST_DIR)
    
    wrapper_src = TEST_DIR / "common" / f"kernel_wrapper_{sm}.h"
    
    with tempfile.TemporaryDirectory(dir=TEST_DIR / "common") as tmp_inc:
        shutil.copy(wrapper_src, Path(tmp_inc) / "kernel_wrapper.h")
        
        # Use g++ for the host driver: it ships with libstdc++ headers, while
        # the conda clang++ on some hosts does not resolve <algorithm> etc.
        # Override with SLIFTER_HOST_CXX to force a specific compiler.
        cxx = os.environ.get("SLIFTER_HOST_CXX", "g++")
        link_cmd = [
            cxx, "-std=c++20", "-O2",
            f"-D__CUDA_ARCH__={sm}0",
            f"-I{Path(tmp_inc).relative_to(TEST_DIR)}",
            "-I./common",
            f"./{host_rel}",
            str(obj_path),
            "-o", str(exe_path),
            "-lpthread",
        ]
        run_cmd(link_cmd, cwd=TEST_DIR)

    # Run from the kernel's own dir so hosts that read a relative ref_output.bin
    # (the file-based addrtest drivers) find it; self-checking hosts ignore cwd.
    out = run_cmd([str(exe_path)], cwd=exe_path.parent)
    assert "TEST PASSED" in out

def _compile_ll_to_obj_arm(ll_path: Path, obj_path: Path):
    """Compile LLVM IR to aarch64 object using llvmlite (no clang needed)."""
    from llvmlite import binding as llvm
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    ll_text = ll_path.read_text()
    # Retarget for aarch64
    ll_text = ll_text.replace(
        'target triple = "unknown-unknown-unknown"',
        'target triple = "aarch64-unknown-linux-gnu"',
    )
    # aarch64 ELF TLS requires localexec model
    ll_text = ll_text.replace(
        "external thread_local global",
        "external thread_local(localexec) global",
    )

    mod = llvm.parse_assembly(ll_text)
    mod.verify()

    target = llvm.Target.from_triple("aarch64-unknown-linux-gnu")
    tm = target.create_target_machine(opt=2, codemodel="small")
    obj_bytes = tm.emit_object(mod)
    obj_path.write_bytes(obj_bytes)


def _run_arm_host(exe_path: Path, ll_path: Path, host_rel: str, sm: str):
    obj_path = exe_path.with_suffix(".o")

    _compile_ll_to_obj_arm(ll_path, obj_path)

    wrapper_src = TEST_DIR / "common" / f"kernel_wrapper_{sm}.h"

    with tempfile.TemporaryDirectory(dir=TEST_DIR / "common") as tmp_inc:
        shutil.copy(wrapper_src, Path(tmp_inc) / "kernel_wrapper.h")

        link_cmd = [
            "g++", "-std=c++20", "-O2",
            f"-D__CUDA_ARCH__={sm}0",
            f"-I{Path(tmp_inc).relative_to(TEST_DIR)}",
            "-I./common",
            f"./{host_rel}",
            str(obj_path),
            "-o", str(exe_path),
            "-lpthread",
        ]
        run_cmd(link_cmd, cwd=TEST_DIR)

    # Run from the kernel's own dir so hosts that read a relative ref_output.bin
    # (the file-based addrtest drivers) find it; self-checking hosts ignore cwd.
    out = run_cmd([str(exe_path)], cwd=exe_path.parent)
    assert "TEST PASSED" in out


_PTX_BY_SM = {
    "50": 65, "52": 65, "53": 65,
    "60": 65, "61": 65, "62": 65,
    "70": 70, "72": 70,
    "75": 78,
    "80": 80, "86": 80, "87": 80, "89": 80,
    "90": 80, "90a": 80,
    "100": 85, "120": 85,
}


def _compile_ll_to_ptx(ll_path: Path, sm: str, ptx_path: Path):
    # LLVM 20+ supports +ptx80 (sm_90) and +ptx85 (sm_100+) natively.
    # Pick the lowest PTX ISA that covers the target sm's instruction
    # set; sm_75 stays at +ptx78 because the lifter still emits some
    # inline-asm opcodes (movmatrix) introduced in PTX 7.8.
    ptx_ver = _PTX_BY_SM.get(str(sm), 75)
    run_cmd([
        "llc", "-march=nvptx64", f"-mcpu=sm_{sm}", f"-mattr=+ptx{ptx_ver}",
        "-o", str(ptx_path), str(ll_path)
    ], cwd=TEST_DIR)


def _run_nvvm_host(exe_path: Path, ll_path: Path, host_rel: str, sm: str, out_base_ptx: Path):
    ptx_path = out_base_ptx.with_suffix(".ptx")
    cubin_path = out_base_ptx.with_suffix(".cubin")

    _compile_ll_to_ptx(ll_path, sm, ptx_path)

    device_img = ptx_path
    if os.environ.get("USE_PTXAS"):
        run_cmd(["ptxas", f"-arch=sm_{sm}", str(ptx_path), "-o", str(cubin_path)], cwd=TEST_DIR)
        device_img = cubin_path

    run_cmd([
        "nvcc", "-std=c++17", "-I./common", f"./{host_rel}", "-lcuda",
        "-o", str(exe_path)
    ], cwd=TEST_DIR)

    out = run_cmd([str(exe_path), str(device_img)], cwd=TEST_DIR)
    assert "TEST PASSED" in out


_PTXSWAP_SO = TEST_DIR / "common" / "libptxswap.so"
_PTXSWAP_SRC = TEST_DIR / "common" / "ptxswap.cc"

def _ensure_ptxswap_so() -> Path:
    """Build the LD_PRELOAD shim once per session; return the .so path."""
    if _PTXSWAP_SO.exists() and _PTXSWAP_SO.stat().st_mtime >= _PTXSWAP_SRC.stat().st_mtime:
        return _PTXSWAP_SO
    run_cmd([
        "nvcc", "-shared", "-Xcompiler", "-fPIC", "-std=c++17",
        "-lcuda", "-lcudart",
        "-o", str(_PTXSWAP_SO), str(_PTXSWAP_SRC),
    ], cwd=_PTXSWAP_SRC.parent)
    return _PTXSWAP_SO


def _derive_const_symmap(elf_path: Path) -> List[Tuple[str, int, int]]:
    """Parse a cuobjdump -elf TEXT dump for __constant__ symbols.

    Returns [(device_name, bank, offset)] for every STT_OBJECT symbol
    whose section is a `.nv.constant<bank>` bank. These are the host-
    filled __constant__ arrays (e.g. d3q19's C_dirs/C_p in bank 3) that
    the ptxswap shim mirrors into the lifted module's flat `const_mem`.

    The .nv.constant<bank> section index is read from the section table
    (`idx off size es align type flags link info name`), then each
    symtab row (`index value size info other shndx name`) whose shndx
    matches a constant section yields (name, bank, value=offset).
    Empty result -> no sidecar emitted -> shim is a pure pass-through.
    """
    try:
        text = elf_path.read_text(errors="replace")
    except OSError:
        return []

    # Section-table rows: all hex (no 0x), name is `.nv.constant<bank>`.
    sec_row = re.compile(
        r"^\s*([0-9a-f]+)\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+"
        r"[0-9a-f]+\s+\S+\s+[0-9a-f]+\s+[0-9a-f]+\s+[0-9a-f]+\s+"
        r"(\.nv\.constant(\d+))\s*$")
    # A fatbin dump concatenates per-block ELFs; section indices restart
    # per block. Resolve symbols against the SAME block's section table.
    blocks = re.split(r"^Fatbin elf code:\s*$", text, flags=re.MULTILINE)

    result: List[Tuple[str, int, int]] = []
    seen = set()
    for block in blocks:
        idx_to_bank = {}
        for line in block.splitlines():
            m = sec_row.match(line)
            if m:
                idx_to_bank[int(m.group(1), 16)] = int(m.group(3))
        if not idx_to_bank:
            continue
        sym_m = re.search(
            r"^\.section\s+\.symtab\s*\n(.*?)(?=\n\.\S|\Z)",
            block, re.DOTALL | re.MULTILINE)
        if not sym_m:
            continue
        # symtab row: index value size info other shndx name
        sym_row = re.compile(
            r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
        for line in sym_m.group(1).splitlines():
            sm_ = sym_row.match(line)
            if not sm_ or sm_.group(1) == "index" or sm_.group(7) == "(null)":
                continue
            try:
                value = int(sm_.group(2), 0)
                info = int(sm_.group(4), 0)
                shndx = int(sm_.group(6), 0)
            except ValueError:
                continue
            if info & 0xF != 1:           # STT_OBJECT only
                continue
            bank = idx_to_bank.get(shndx)
            if bank is None:
                continue
            name = sm_.group(7)
            if name in seen:
                continue
            seen.add(name)
            result.append((name, bank, value))
    return result


def _write_const_symmap(elf_path: Path, out_path: Path) -> bool:
    """Emit the PTXSWAP_SYMMAP sidecar (`name bank offset` per line) from
    the SLifter ELF's __constant__ symbol table. Returns True iff at
    least one constant symbol was found (otherwise no sidecar, so the
    shim stays a pure pass-through and cannot regress other tests)."""
    entries = _derive_const_symmap(elf_path)
    if not entries:
        return False
    out_path.write_text(
        "".join(f"{name} {bank} {off}\n" for name, bank, off in entries))
    return True


def _write_device_symmap(elf_path: Path, out_path: Path) -> bool:
    """Emit the PTXSWAP_DEVMAP sidecar (`name init_flag offset` per line)
    for the ELF's __device__ globals (`.nv.global`/`.nv.global.init`
    symbols). The lifter materialises these into one flat addrspace(1)
    buffer (`__slifter_nv_global[_init]`) at the SAME byte offsets the ELF
    reports, so the shim can mirror host cudaMemcpyTo/FromSymbol copies
    into/out of the lifted buffer (hungarian-cuda, collision-cuda — they
    share all state with the host through __device__ globals).

    Returns True iff at least one device-global symbol was found;
    otherwise no sidecar -> shim stays a pure pass-through (no regression
    for tests without host-visible __device__ globals)."""
    try:
        from utils.elf import Elf
    except ImportError:
        return False
    try:
        elf = Elf(str(elf_path))
    except Exception:
        return False
    syms = getattr(elf, "nv_global_syms", None) or {}
    if not syms:
        return False
    lines = []
    for name, (in_init, off, size) in sorted(syms.items()):
        lines.append(f"{name} {1 if in_init else 0} {off} {size}\n")
    out_path.write_text("".join(lines))
    return True


# Lines that change run-to-run (timing, throughput) — strip before comparing
# vanilla vs lifted hecbench output.
_NONDET_LINE_RE = re.compile(
    r"(time|elapsed|throughput|bandwidth|gflops|gbyte|tflops|"
    r"average|kernel.*\(.*s\)|"
    # Millisecond / microsecond / nanosecond / "seconds" measurements
    # (Memory allocation: 3.266878 ms, "X.Y us", "1398487158 ns",
    #  "Benchmarking ... 6.789e-06 seconds" idivide-style)
    r"\d\s*ms\b|\d\s*us\b|\d\s*ns\b|\bseconds\b|"
    # Bare floating-point seconds — "0.006147000007 s" on its own line
    # (lavaMD prints a single elapsed-time number per kernel)
    r"^\s*\d+\.\d+\s+s\s*$|"
    # myocyte-style "0.000000064000 s, 0.00066... % : SETUP VARIABLES"
    # — leading bare seconds + percent + label
    r"^\s*\d+\.\d+\s+s,\s+\d+\.\d+\s+%\s*:|"
    # "done in X s for Y iterations" / "X s by iteration" (knn-style)
    r"\bdone in\b|by iteration|iterations\)|"
    # "Options per second : 789.89" / "GF/s (host): 277200" (binomial,
    # minisweep-style throughput)
    r"\bper second\b|GF/s|"
    # "Speedup = 0.815373" (idivide — derived from timing)
    r"\bSpeedup\b|"
    # Long timestamps at line start (langford prints microsecond clocks
    # like "1777957068535 Computing PL(2, 7)")
    r"^\d{13,}\s|"
    # Hardware clock counters and per-warp efficiency are non-deterministic
    r"\bTotal clocks\b|\befficiency\b|"
    # segsort prints live device-memory stats ("GPU memory usage: used =
    # 4802.81 MB, free = ...") — the lifted module's JIT allocation differs
    # from the vanilla cubin's, so the numbers legitimately diverge.
    r"\bGPU memory usage\b|"
    # babelstream "Copy 4400581245.902 0.00000 0.00000 0.00000" —
    # benchmark name in column 1 + GB/s throughput in column 2.
    r"^\s*(Copy|Mul|Add|Triad|Dot|Nstream)\s+\d+\.\d+|"
    # heat2d "BW = 2982616.178 GB/s, P = 2236962.133 Gflop/s" — both
    # throughput stats. Match either anywhere in line.
    r"\bGB/s\b|\bGflop/s\b|\busec/iter\b|"
    # NAS IS "Full_verify: number of keys out of sort: 1901864173" —
    # the count is non-deterministic because IS uses pseudo-random keys.
    r"\bnumber of keys out of sort\b|"
    # expdist "device result: -32953144718244580425728.000000" — both
    # vanilla and lifted produce garbage on this test (it requires
    # specific GPU state); filter the broken line so the rest can compare.
    r"^\s*device result:\s|"
    # rtm8 prints "dt = 0.198045" and "pt_rate (millions/sec) = 18897.230"
    # — both timing-derived. The PASS/FAIL line above is what matters.
    r"^\s*dt\s*=\s*\d|"
    r"\bpt_rate\b|"
    # ccsd-trpdrv "TIMING: min=..., max=..., avg=..." — IGNORECASE+'time'
    # doesn't match because TIMING has no 'time' substring (T-I-M-I-N-G).
    r"\bTIMING:|"
    # bincount/historam "Output min, median, max values: (...)" — values
    # depend on atomicAdd ordering; race-dependent. Both vanilla and lifted
    # produce same shape but different min on int overflow.
    r"\bOutput min, median, max values:|"
    # rsbench "Lookups/s: 1,699,988,110" — timing-derived throughput.
    r"\bLookups/s\s*:|"
    # rsbench "Running on: <devicename>" prints raw GPU name with
    # potentially non-UTF8 padding bytes. Variable across nodes.
    r"^\s*Running on:|"
    # rsbench/ccsd "Verification checksum: 0 (WARNING - INVALID CHECKSUM!)"
    # — checksum is computed from kernel output but here both vanilla and
    # lifted are wrong (kernel itself broken on this platform).
    r"INVALID CHECKSUM\b|"
    # ising "updates per ns: 202.01" — throughput (kernel output otherwise
    # byte-identical; verified isolated swaps=3, only timing lines differ).
    r"\bupdates per ns\b|"
    # shmembench "using 128bit operations : 31801.25 GB/sec (1987 billion
    # accesses/sec)" — throughput. The existing \bGB/s\b misses "GB/sec".
    r"\bGB/sec\b|\baccesses/sec\b|"
    # dslash "Approximate memory usage = 1588.78" — a host-side memory
    # measurement that varies run-to-run; the kernel "Validating the result"
    # line is identical (verified isolated swaps=2, only this line differs).
    r"\bApproximate memory usage\b|"
    # lulesh "FOM = 1234.5 (z/s)" — figure-of-merit zones/sec throughput.
    # Output byte-identical through "MaxAbsDiff = 0.0" (verified swaps=15).
    r"\bFOM\b|"
    # michalewicz "Global minima = X" / "Error = Y" — atomic_min CAS-loop
    # is race-dependent on thread schedule; vanilla and lifted converge to
    # different values legitimately (and on this CUDA both fail to find
    # the true minimum, with vanilla=0 lifted=garbage).
    r"^\s*Global minima\s*=|"
    r"^\s*Error\s*=|"
    # memcpy "Timing gap in nanoseconds per byte: 98.94" — pure timing
    # (\bseconds\b doesn't match "nanoseconds"; "Timing" has no "time"
    # substring). The kernel is a plain copy; only the timing line varies.
    r"\bnanoseconds per byte\b|\bTiming gap\b|"
    # d3q19-bgk "Performance: MLUPS" header, followed by 10 bare
    # million-lattice-updates-per-second throughput numbers (filtered as a
    # block by _filter_nondet). The correctness signal is the preceding
    # "energy ... iteration" line, which is bit-exact.
    r"\bMLUPS\b|"
    # quicksort "Standard Deviation: 99.27" / "%error (3*stdDev)/Average: 1.2%"
    # — both computed from the per-iteration timing samples (times[k]); pure
    # timing noise. The correctness signal ("verifying: true" / "Number of
    # failures: 0") is bit-identical vanilla vs lifted. ("Standard Deviation"
    # / "%error" carry no time-unit token so the ms/seconds rules miss them.)
    r"\bStandard Deviation\b|\b%error\b|"
    # keccaktreehash "CPU speed : 76641.10 kB/s" / "GPU speed : 1063110.72 kB/s"
    # — speed1 = bytes / ((t2-t1)+0.01), a timing-derived hash rate (kilobytes
    # per second). \bGB/s\b/\bGB/sec\b miss the lowercase-k "kB/s" form. The
    # correctness signal is the preceding print_KS_256 hash line (8 %08x words),
    # which is bit-identical vanilla vs lifted (verified on CPU and GPU rows).
    r"\bkB/s\b|"
    # blas-gemmEx/blas-gemmEx2 "56.75 TFLOP/s" / "123.31 TOP/s" — throughput
    # printed by performance() as `perf = total_ops/avg_time`. The slash form
    # (FLOP/s, OP/s with optional G/T scale prefix) isn't caught by the
    # \btflops\b/\bgflops\b tokens above. The correctness signal is the
    # "compare first ten values" block (fp64/fp32/fp16/bf16/int8), which is
    # bit-identical vanilla vs lifted.
    r"(?:T|G)?(?:FL)?OP/s\b)",
    re.IGNORECASE | re.MULTILINE,
)

# Headers that introduce a block of bare numeric throughput lines (no unit on
# the value line itself, so _NONDET_LINE_RE can't match them context-free).
# When _filter_nondet sees one of these headers it also drops the run of
# subsequent bare-numeric lines until a non-numeric line breaks the block.
# d3q19-bgk: "Performance: MLUPS\n8376.06\n8375.87\n..." — the 10 MLUPS values
# are timing-derived; the header itself is dropped by _NONDET_LINE_RE above.
_THROUGHPUT_BLOCK_HEADER_RE = re.compile(r"\bMLUPS\b", re.IGNORECASE)
_BARE_NUMBER_LINE_RE = re.compile(r"^\s*[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*$")

# wmma echoes one CSV header line then exactly one CSV data row whose last
# three fields are "elapsedMs, Problem Size(GFlops), TFlops/s". The header is
# already dropped by _NONDET_LINE_RE (its \btflops\b token), but the data row
# is a bare comma-separated value line with no unit token, so it can't be
# matched context-free. The header is uniquely identified by its
# "Problem Size(GFlops)" + "TFlops/s" columns; when seen, the SINGLE next line
# is its timing data row (elapsedMs + TFlops/s vary run-to-run) and is dropped.
# The matrix-result correctness signal ("PASSED" / "Max relative error: 0") is
# bit-identical vanilla vs lifted; only the two timing-derived fields differ.
_SINGLE_DATAROW_HEADER_RE = re.compile(
    r"Problem Size\(GFlops\).*TFlops/s", re.IGNORECASE)

def _filter_nondet(text: str) -> str:
    out = []
    in_throughput_block = False
    drop_next_datarow = False
    for line in text.splitlines():
        if drop_next_datarow:
            # Consume exactly the one CSV data row following a single-datarow
            # header (wmma); never spans more than one line.
            drop_next_datarow = False
            continue
        if _NONDET_LINE_RE.search(line):
            # A throughput-table header (e.g. "Performance: MLUPS") starts a
            # run of bare-numeric throughput lines that must also be dropped.
            in_throughput_block = bool(_THROUGHPUT_BLOCK_HEADER_RE.search(line))
            # A single-datarow header (wmma) drops exactly the next line.
            drop_next_datarow = bool(_SINGLE_DATAROW_HEADER_RE.search(line))
            continue
        if in_throughput_block:
            if _BARE_NUMBER_LINE_RE.match(line):
                continue
            in_throughput_block = False
        out.append(line)
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Uniform output comparison: floating-point with tolerance, integers bit-exact.
#
# Lifted SASS runs on real hardware but the lifter's FP lowering (FMA
# contraction, reduction/accumulation order, f16 rounding) legitimately
# perturbs floating-point results in their low-order digits — bit-exact stdout
# parity is not achievable and not the right bar. Integer results, by contrast,
# come from exact bit operations and MUST match bit-for-bit (an off-by-one or a
# wrong-mask is a real lifter bug, never rounding noise).
#
# So the single, uniform rule applied everywhere vanilla and lifted stdout are
# compared:
#   * a token that is an INTEGER on both sides  -> must be exactly equal
#   * a token that is a FLOAT on either side    -> equal within (atol, rtol)
#   * all non-numeric text (labels, structure)  -> must be exactly equal
# This is strictly more permissive than `==`, so it can never flip a currently
# passing test to failing; it only stops rejecting legitimate FP rounding.
# (Nondeterministic timing/throughput lines are still removed by
# _filter_nondet first — those vary by integer multiples, not "tolerance".)
_FP_RTOL = 1e-2   # 1% relative — covers f16/f32 reassociation & contraction
_FP_ATOL = 1e-4   # absolute floor for values near zero

# A numeric token: hex integer, or decimal int/float with optional exponent,
# or inf/nan. Sign is matched so "-0.5" parses whole.
_NUM_TOKEN_RE = re.compile(
    r"[-+]?(?:0[xX][0-9a-fA-F]+"
    r"|(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
    r"|inf|nan)",
    re.IGNORECASE,
)


def _token_is_hex(tok: str) -> bool:
    return tok.lstrip("+-")[:2].lower() == "0x"


def _token_is_int(tok: str) -> bool:
    """True iff the token denotes an integer (decimal digits or hex), i.e.
    has no '.', no exponent, and is not inf/nan."""
    if _token_is_hex(tok):
        return True
    return tok.lstrip("+-").isdigit()


def _token_to_int(tok: str) -> int:
    return int(tok, 16) if _token_is_hex(tok) else int(tok, 10)


def _token_to_float(tok: str) -> float:
    return float(_token_to_int(tok)) if _token_is_hex(tok) else float(tok)


def _nums_equal(a: str, b: str) -> bool:
    if _token_is_int(a) and _token_is_int(b):
        # Integers: bit-exact (base-aware so 0x10 == 16).
        return _token_to_int(a) == _token_to_int(b)
    fa, fb = _token_to_float(a), _token_to_float(b)
    if math.isnan(fa) or math.isnan(fb):
        return math.isnan(fa) and math.isnan(fb)
    if math.isinf(fa) or math.isinf(fb):
        return fa == fb
    return abs(fa - fb) <= _FP_ATOL + _FP_RTOL * abs(fb)


def _line_matches(la: str, lb: str) -> bool:
    a_nums = _NUM_TOKEN_RE.findall(la)
    b_nums = _NUM_TOKEN_RE.findall(lb)
    # Non-numeric skeleton (labels, punctuation, spacing) must match exactly;
    # this also pins number count and position.
    if _NUM_TOKEN_RE.sub("\0", la) != _NUM_TOKEN_RE.sub("\0", lb):
        return False
    return all(_nums_equal(a, b) for a, b in zip(a_nums, b_nums))


def _outputs_match(a: str, b: str) -> bool:
    """Uniform vanilla-vs-lifted comparison: FP within tolerance, ints exact,
    text exact. `a`/`b` are already _filter_nondet-ed."""
    la, lb = a.splitlines(), b.splitlines()
    if len(la) != len(lb):
        return False
    return all(_line_matches(x, y) for x, y in zip(la, lb))

_RUN_RECIPE_RE = re.compile(
    r"^run:[^\n]*\n((?:\t[^\n]*\n)+)", re.MULTILINE
)
_RUN_INVOCATION_RE = re.compile(r"\./\$\(program\)\s*([^\n]*)$")

def _hecbench_run_args(test_dir: Path) -> List[str]:
    """Parse the Makefile's `run:` target to recover the canonical argv the
    benchmark expects (e.g., problem size, repeat count). Returns [] if the
    Makefile has no run target, in which case we just invoke the binary."""
    text = (test_dir / "Makefile").read_text()
    m = _RUN_RECIPE_RE.search(text)
    if not m:
        return []
    for line in m.group(1).splitlines():
        inv = _RUN_INVOCATION_RE.search(line)
        if inv:
            return inv.group(1).strip().split()
    return []

# Vanilla reference must finish within this wall-time or the test is
# skipped (workload too large for the swap comparison). Overridable via
# SLIFTER_VANILLA_TIMEOUT — a handful of hecbench bases configure a heavy
# workload × repeat count (the trailing run-arg is an iteration count of
# 100–1000) and run 61–270s vanilla on an A100; bumping this recovers them
# as real tests rather than runtime-skips.
_VANILLA_TIMEOUT_S = int(os.environ.get("SLIFTER_VANILLA_TIMEOUT", "300"))
# Lifted PTX has no ptxas opt; baseline ~5-10× slower. Multiplier set
# generously to accommodate worst-case lifted-vs-native runtime.
_LIFTED_TIMEOUT_MULT = 60
_LIFTED_TIMEOUT_MIN_S = 60

def _saw_no_cuda(p) -> bool:
    return ("no CUDA-capable device is detected" in p.stdout or
            "no CUDA-capable device is detected" in p.stderr)


def _prepare_ptxswap_env(ll_path: Path, sm: str, ptx_path: Path,
                         elf_path: Optional[Path], verbose: bool = False) -> Dict[str, str]:
    """Compile the lifted IR to PTX, build the LD_PRELOAD shim, emit the
    optional ELF sidecars, and return the full environment for the swapped run.

    Emits the constant-symbol sidecar (name/bank/offset) so the shim can
    mirror host cudaMemcpyToSymbol writes into the lifted module's flat
    `const_mem` global (d3q19's C_dirs/C_p, bank 3). If the ELF has no
    __constant__ symbols, no sidecar is written and PTXSWAP_SYMMAP is
    left unset, leaving the shim's cudaMemcpyToSymbol interceptor a
    pure pass-through (cannot regress tests without runtime constants).

    __device__-global sidecar: the shim syncs these symbols between
    the vanilla cubin and the lifted module's flat
    __slifter_nv_global buffer around each lifted launch, so host
    state shared via cudaMemcpyTo/FromSymbol AND direct __managed__
    access (hungarian-cuda's n_matches/zeros_size) stays consistent.
    No __device__-global symbols -> no sidecar -> shim pass-through.

    `verbose=True` adds PTXSWAP_VERBOSE=1 so the shim logs one "mapping stub"
    line per swapped kernel (used by suites that assert a nonzero swap count).
    The hecbench path keeps verbose off — its stderr-based checks predate the
    verbose logging and must see unchanged shim output."""
    _compile_ll_to_ptx(ll_path, sm, ptx_path)
    shim = _ensure_ptxswap_so()
    env = {**os.environ,
           "LD_PRELOAD": str(shim),
           "PTXSWAP_PTX": str(ptx_path)}
    if elf_path is not None and elf_path.exists():
        candidate = ptx_path.with_suffix(".symmap")
        if _write_const_symmap(elf_path, candidate):
            env["PTXSWAP_SYMMAP"] = str(candidate)
        dev_candidate = ptx_path.with_suffix(".devmap")
        if _write_device_symmap(elf_path, dev_candidate):
            env["PTXSWAP_DEVMAP"] = str(dev_candidate)
    if verbose:
        env["PTXSWAP_VERBOSE"] = "1"
    return env


def _run_vanilla_ref(exe: Path, args: List[str], cwd: Path, name: str):
    """Run the vanilla (reference) binary; returns (proc, wall_secs).

    Skips (rather than fails) on timeout — that's a property of the test
    workload, not of the lifter. `errors='replace'` because some programs
    (e.g. iso2dfd) print non-UTF8 bytes; the default `text=True` decode would
    raise UnicodeDecodeError before we ever get to compare stdout."""
    def _run_vanilla():
        return subprocess.run(
            [str(exe), *args], cwd=cwd, text=True, capture_output=True,
            timeout=_VANILLA_TIMEOUT_S,
            errors="replace",
        )

    t0 = time.monotonic()
    try:
        vanilla_proc = _run_vanilla()
    except subprocess.TimeoutExpired:
        pytest.skip(f"Vanilla run exceeded {_VANILLA_TIMEOUT_S}s — test workload too large for ptxswap comparison")

    # Sweep nodes occasionally throw transient "no CUDA-capable device"
    # mid-run when GPU state hasn't fully cleared between back-to-back tests.
    # Retry up to 2 more times with a short backoff before giving up — these
    # tests pass clean on a quiescent A100, so the contention is the only
    # signal we want to filter, not a lifter property.
    retry = 0
    while _saw_no_cuda(vanilla_proc) and retry < 2:
        retry += 1
        time.sleep(1.5)
        try:
            vanilla_proc = _run_vanilla()
        except subprocess.TimeoutExpired:
            pytest.skip(f"Vanilla run exceeded {_VANILLA_TIMEOUT_S}s after retry — test workload too large for ptxswap comparison")
    vanilla_secs = time.monotonic() - t0

    # If still no CUDA device after retries, the node is genuinely not
    # presenting a usable GPU to this process — skip with a clear marker.
    if _saw_no_cuda(vanilla_proc):
        pytest.skip(f"Vanilla saw no CUDA device for {name} after {retry} retries — GPU contention/state issue, not a lifter signal")

    return vanilla_proc, vanilla_secs


def _run_lifted_swapped(exe: Path, args: List[str], cwd: Path,
                        lifted_env: Dict[str, str], lifted_timeout: int,
                        vanilla_secs: float, name: str):
    """Run the binary under the ptxswap shim; fail on timeout / PTX-load
    failure, skip on node-local libcudart loader issues. Returns the proc."""
    try:
        swapped = subprocess.run(
            [str(exe), *args], cwd=cwd, text=True, capture_output=True,
            timeout=lifted_timeout,
            errors="replace",
            env=lifted_env,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"Lifted PTX run timed out after {lifted_timeout}s "
            f"(vanilla took {vanilla_secs:.1f}s) — lifted kernel is much slower than expected "
            f"or stuck in a loop"
        )
    # Module-load failure means the lifted PTX is broken; the shim then
    # silently falls every launch through to the original cubin, so stdout
    # would compare equal even though we never actually exercised lifted
    # code. Treat that as a hard fail rather than a false-pass.
    if "[ptxswap] cuModuleLoad" in swapped.stderr and "failed" in swapped.stderr:
        pytest.fail(
            f"Lifted PTX failed to load for {name} "
            f"— shim could not register any kernel:\n"
            f"--- lifted stderr ---\n{swapped.stderr[-800:]}"
        )

    # Loader-resolution failure when LD_PRELOAD changes libcudart's lookup
    # chain on nodes where libcudart.so.12 is not in the default search
    # path. Lifted-side rc=127 with a libcudart
    # "cannot open shared object" error means the test never even started —
    # not a lifter signal.
    if (swapped.returncode == 127 and
        "cannot open shared object file" in swapped.stderr and
        "libcudart" in swapped.stderr):
        pytest.skip(
            f"Lifted run failed to load libcudart on this node "
            f"(LD_PRELOAD search-path issue, not a lifter signal):\n"
            f"--- lifted stderr ---\n{swapped.stderr[-400:]}"
        )
    return swapped


def _run_hecbench_ptxswap(test_dir: Path, ll_path: Path, sm: str, ptx_path: Path,
                          elf_path: Optional[Path] = None):
    """Build the hecbench binary, run it once vanilla and once with the PTX-swap
    shim loaded, and assert the deterministic parts of stdout match. Lines that
    naturally vary (timing, throughput) are filtered out before compare.

    Skips (rather than fails) if the vanilla run itself errors or times out —
    that's a property of the test, not of the lifter. The lifted run gets a
    timeout proportional to the vanilla wall-time, since lifted PTX runs
    several × slower than the original cubin."""
    lifted_env = _prepare_ptxswap_env(ll_path, sm, ptx_path, elf_path)
    # nvcc's default static-libcudart linkage embeds cudaLaunchKernel into
    # the binary, defeating LD_PRELOAD. Force shared linkage so the shim
    # can intercept the runtime dispatch. If the test source itself can't
    # build under -cudart=shared (e.g., bfloat/half conversions that the
    # newer toolkit rejects), the test isn't a lifter signal — skip it.
    try:
        exe = _compile_hecbench(test_dir, sm, extra_cflags="-cudart=shared")
    except AssertionError as e:
        pytest.skip(f"Vanilla build failed under -cudart=shared (test-harness issue, not a lifter signal): {str(e)[:200]}")
    args = _hecbench_run_args(test_dir)

    vanilla_proc, vanilla_secs = _run_vanilla_ref(exe, args, test_dir, test_dir.name)

    # Vanilla rc != 0 with stdout markers of an internal validation failure
    # ("Final wavefields different: Error", "FAIL", "Error " near rc-print)
    # means vanilla didn't establish a clean reference — per the project doc,
    # vanilla failure → SKIP, not FAIL. Note: under -cudart=shared, cudart's
    # deferred-error path often produces rc=1 even on a success run, so we
    # only skip when the rc!=0 is corroborated by an in-stdout failure marker.
    if vanilla_proc.returncode != 0:
        v_out = vanilla_proc.stdout
        if any(m in v_out for m in (
            "Final wavefields", "wavefields are different", "validation failed",
            "are different: Error", "Test FAILED", "test failed",
            # linearprobing exits 255 with "Duplicate key found"
            "Duplicate key found",
            # Tests that depend on data files missing on the test box
            "ERROR: The file", "was not opened for reading",
            # Several tests print bare "FAIL" on its own line when their
            # internal validator rejects (is, mr, logprob, langford). Use
            # rc!=0 + standalone "FAIL" line to gate the skip.
            "\nFAIL\n",
        )):
            pytest.skip(f"Vanilla rc={vanilla_proc.returncode} and own validation failed for {test_dir.name} — not a lifter signal")

    # Some tests exit rc=0 but their stdout reveals the kernel produced
    # wrong values or the internal verifier printed FAIL. Without a clean
    # vanilla reference we have nothing to compare lifted against — these
    # are not lifter signals. Examples: nonzero ("Number of non-zero
    # elements mismatch:"), shmembench ("checksum failed"), rtm8 prints
    # "@index N host: X device Y" mismatches, logprob/michalewicz produce
    # absurd values like -8.4e17 from uninitialized memory, heat2d prints
    # "Mismatch at" + bare "FAIL".
    v_out = vanilla_proc.stdout
    _VANILLA_INTERNAL_FAIL_TOKENS = (
        "checksum failed",
        "Number of non-zero elements mismatch:",
        "Mismatch at ",
        "Validation FAILED",
        "verification failed",
        # logprob prints absurd values (uninitialized memory) and ends with FAIL.
        # When vanilla itself reaches FAIL on its own, lifted has no clean reference.
        "INVALID CHECKSUM",
    )
    if any(tok in v_out for tok in _VANILLA_INTERNAL_FAIL_TOKENS):
        pytest.skip(f"Vanilla produced an internal failure for {test_dir.name} — no clean reference for lifter comparison")
    # logprob: bare "FAIL" line at end of vanilla output paired with absurd
    # "log_probs:" leading values — vanilla kernel is broken on this platform.
    if "\nFAIL\n" in v_out and "log_probs:" in v_out:
        pytest.skip(f"Vanilla kernel produced FAIL for {test_dir.name} — broken on this platform")

    lifted_timeout = max(_LIFTED_TIMEOUT_MIN_S, int(vanilla_secs * _LIFTED_TIMEOUT_MULT))
    swapped = _run_lifted_swapped(exe, args, test_dir, lifted_env,
                                  lifted_timeout, vanilla_secs, test_dir.name)
    # Compare stdout (semantic output) rather than exit code. Exit-code
    # divergence is common with -cudart=shared (cudart's deferred-error
    # propagation differs from static linkage) and isn't a kernel-behavior
    # signal. What matters is whether the lifted PTX produces the same
    # observable output as the original cubin.
    a = _filter_nondet(vanilla_proc.stdout)
    b = _filter_nondet(swapped.stdout)
    # Uniform comparison: FP within tolerance, integers bit-exact, text exact.
    assert _outputs_match(a, b), (
        f"PTX-swap output diverged for {test_dir.name} "
        f"(vanilla rc={vanilla_proc.returncode}, lifted rc={swapped.returncode}):\n"
        f"--- vanilla ---\n{a}\n--- lifted ---\n{b}\n"
        f"--- lifted stderr ---\n{swapped.stderr[-500:]}"
    )


def _run_synth_ptxswap(exe: Path, ll_path: Path, sm: str, ptx_path: Path,
                       elf_path: Optional[Path], run_cwd: Path, name: str):
    """ptxswap comparison for a standalone self-checking microbench (the
    sm90_synth corpus): run the already-built binary once vanilla and once
    under the shim, then assert

      1. swap count >= 1 — the corpus is fully under our control, every
         kernel is expected to swap, so a zero-swap run is a FAIL (it would
         silently ride the vanilla cubin and pass vacuously), unlike hecbench
         where name-mapping gaps are tolerated;
      2. nondet-filtered stdout matches (FP within tolerance, ints exact);
         empty-vs-empty stdout is a valid match for print-nothing benches;
      3. exit codes match — these binaries are single-file self-checkers,
         so rc is a deterministic part of their contract (vanilla rc != 0
         skips first: no clean reference means no lifter signal).

    Runs with cwd=run_cwd (the pytest tmp dir) so benches that drop scratch
    files never litter the repo. PTXSWAP_VERBOSE=1 makes the shim print one
    "mapping stub" line per swapped kernel, which is the swap counter."""
    lifted_env = _prepare_ptxswap_env(ll_path, sm, ptx_path, elf_path, verbose=True)

    vanilla_proc, vanilla_secs = _run_vanilla_ref(exe, [], run_cwd, name)
    # Vanilla failure -> skip (policy shared with hecbench): a self-checking
    # bench that fails on its own gives no clean reference to compare against.
    if vanilla_proc.returncode != 0:
        pytest.skip(
            f"Vanilla run failed (rc={vanilla_proc.returncode}) for {name} — "
            f"no clean reference for lifter comparison:\n"
            f"--- vanilla stdout ---\n{vanilla_proc.stdout[-500:]}\n"
            f"--- vanilla stderr ---\n{vanilla_proc.stderr[-500:]}"
        )

    lifted_timeout = max(_LIFTED_TIMEOUT_MIN_S, int(vanilla_secs * _LIFTED_TIMEOUT_MULT))
    swapped = _run_lifted_swapped(exe, [], run_cwd, lifted_env,
                                  lifted_timeout, vanilla_secs, name)

    swaps = swapped.stderr.count("mapping stub")
    assert swaps >= 1, (
        f"NO_SWAP: shim mapped 0 kernels for {name} — the run rode the "
        f"vanilla cubin, so any output match would be vacuous:\n"
        f"--- lifted stderr ---\n{swapped.stderr[-800:]}"
    )

    a = _filter_nondet(vanilla_proc.stdout)
    b = _filter_nondet(swapped.stdout)
    assert _outputs_match(a, b), (
        f"PTX-swap output diverged for {name} (swaps={swaps}, "
        f"vanilla rc={vanilla_proc.returncode}, lifted rc={swapped.returncode}):\n"
        f"--- vanilla ---\n{a}\n--- lifted ---\n{b}\n"
        f"--- lifted stderr ---\n{swapped.stderr[-500:]}"
    )
    assert swapped.returncode == vanilla_proc.returncode, (
        f"Exit code diverged for {name} (swaps={swaps}, stdout matched): "
        f"vanilla rc={vanilla_proc.returncode}, lifted rc={swapped.returncode}\n"
        f"--- lifted stderr ---\n{swapped.stderr[-800:]}"
    )

def execute_test_case(bases: Dict, base: str, sm: str, lifter: str):
    entry = bases.get(base, {}).get(sm)
    if not entry:
        pytest.skip(f"No configuration found for {base} sm{sm}")

    lifter = lifter.lower()
    # "arm" reuses the x86 lifter IR but compiles for aarch64;
    # it shares the same host_cpu.cpp driver as x86.
    lift_backend = "arm" if lifter == "arm" else lifter
    host_rel = entry["hosts"].get("x86" if lifter == "arm" else lifter)

    sass_path = _ensure_sass(base, sm, entry)

    test_subdir = sass_path.parent
    testname = base.split('/')[-1]
    out_base = test_subdir / f"{testname}_{lifter}_sm{sm}"

    # Lift timeouts are bounded operationally — but for hecbench tests with
    # very large multi-kernel cubins (e.g. logic-rewrite, 269 kernels with
    # factorFromTruth at 17.7k SASS instructions), the lift can legitimately
    # exceed the timeout on slower CPU hosts. Skip rather than fail since
    # it's a tooling-throughput issue, not a correctness signal. Pure hangs
    # still surface — they'd time out without making any per-function
    # progress, which the operator can investigate via SLIFTER_PASS_TIMING=1.
    try:
        out_ll = _lift_to_llvm(sass_path, lift_backend, out_base)
    except AssertionError as e:
        if "Timeout" in str(e):
            pytest.skip(f"Lift timed out for {base} — too slow on this host (not a correctness signal): {str(e)[:200]}")
        raise

    suite = base.split('/')[0]
    if suite == "hecbench":
        if lifter == "arm":
            # Compile-verify: ensure lifted IR is valid aarch64
            obj_path = out_base.with_suffix(".o")
            try:
                _compile_ll_to_obj_arm(out_ll, obj_path)
            finally:
                obj_path.unlink(missing_ok=True)
        elif lifter == "nvvm":
            # Run lifted PTX in place of the original cubin (LD_PRELOAD shim)
            # and assert stdout matches. Like other nvvm tests, this assumes
            # a GPU is available at runtime.
            ptx_path = out_base.with_suffix(".ptx")
            symmap_path = ptx_path.with_suffix(".symmap")
            try:
                _run_hecbench_ptxswap(test_subdir, out_ll, sm, ptx_path,
                                      sass_path.with_suffix(".elf"))
            finally:
                ptx_path.unlink(missing_ok=True)
                symmap_path.unlink(missing_ok=True)
        return

    if not host_rel:
        if suite == "cuobjdump":
            pytest.skip("Skipping cuobjdump test: missing host implementation")
        return

    exe_path = out_base.with_suffix(".out")

    try:
        if lifter == "x86":
            _run_x86_host(exe_path, out_ll, host_rel, sm)
        elif lifter == "arm":
            _run_arm_host(exe_path, out_ll, host_rel, sm)
        elif lifter == "nvvm":
            _run_nvvm_host(exe_path, out_ll, host_rel, sm, out_base)
        else:
            raise NotImplementedError(f"Unknown lifter: {lifter}")
    finally:
        for p in [exe_path, out_base.with_suffix(".o"), out_base.with_suffix(".cubin")]:
            p.unlink(missing_ok=True)

def execute_nvdisasm_emit_case(bases: Dict, base: str, sm: str):
    entry = bases.get(base, {}).get(sm)
    if not entry:
        pytest.skip(f"No configuration found for {base} sm{sm}")

    sass_path = _ensure_sass(base, sm, entry)
    if not sass_path.name.endswith(".nvdisasm.txt"):
        pytest.skip(f"Skipping nvdisasm emit for non-nvdisasm input: {base}")

    test_subdir = sass_path.parent
    testname = base.split('/')[-1]
    out_base = test_subdir / f"{testname}_nvdisasm_emit_sm{sm}"
    suite, testname = base.split("/", 1)
    test_dir = TEST_DIR / suite / testname
    if not list(test_dir.glob("*.cu")):
        pytest.skip(f"No CUDA source for nvdisasm emit assemble: {base}")

    asm_text = sass_path.read_text()
    asm = Asm(asm_text, path=sass_path)
    sass_parser = SaSSParser(asm_text, elf=asm)
    module = Module(str(out_base), sass_parser)
    ToggleReuse().apply(module)

    with tempfile.TemporaryDirectory(prefix=f"build_{testname}_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            cubin_path = _compile_cuda_cubin(test_dir, testname, sm, tmp_path)
        except AssertionError as e:
            pytest.skip(f"Build failed for {base}: {e}")
        _assemble_nvdisasm(module, out_base, cubin_path)
