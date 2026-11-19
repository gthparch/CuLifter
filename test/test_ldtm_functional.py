import ctypes as ct
import os
import re
import shutil
import sys

import pytest
from llvmlite import binding as llvm

from .test_utils import ROOT, TEST_DIR, run_cmd


FIXTURES = TEST_DIR / "sm100_optest"
LOAD_WIDTHS = re.compile(r"\bLDTM\.x(\d+)\b", re.IGNORECASE)


@pytest.fixture(scope="module")
def ldtm_cubins(tmp_path_factory):
    """Wrap both load routines in the CTA that owns and initializes their TMEM."""
    work = tmp_path_factory.mktemp("ldtm")
    if not all(shutil.which(tool) for tool in ("ptxas", "cuobjdump")):
        pytest.skip("ptxas and cuobjdump are required")
    if "sm_100a" not in run_cmd(["ptxas", "--help"], work):
        pytest.skip("ptxas lacks sm_100a")

    probe = FIXTURES / "ldtm_probe.ptx"
    native = work / "probe.cubin"
    run_cmd(["ptxas", "-arch=sm_100a", probe, "-o", native], work)
    sass = work / "probe.sass"
    sass.write_text(run_cmd(["cuobjdump", "--dump-sass", native], work))
    assert sorted(LOAD_WIDTHS.findall(sass.read_text())) == ["2", "4"]
    run_cmd([sys.executable, ROOT / "main.py", "-i", sass, native,
             "-o", work / "lifted", "--lifter", "nvvm"], work)
    ir = (work / "lifted.ll").read_text()
    assert "sass_unsupported_" not in ir
    module = llvm.parse_assembly(ir)
    module.verify()
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    target = llvm.Target.from_triple(module.triple)
    machine = target.create_target_machine(cpu="sm_100a", features="+ptx87")
    lifted = machine.emit_assembly(module)
    wrapper = (FIXTURES / "ldtm_warp.ptx.inc").read_text()
    cubins = {}
    for name, body in (("native", probe.read_text()), ("lifted", lifted)):
        entry = ".visible .entry read_tmem("
        assert body.count(entry) == 1
        ptx = work / f"{name}_warp.ptx"
        ptx.write_text(body.replace(entry, ".func read_tmem(") + "\n" + wrapper)
        cubin = ptx.with_suffix(".cubin")
        run_cmd(["ptxas", "-arch=sm_100a", ptx, "-o", cubin], work)
        cubins[name] = cubin
    return cubins


def test_ldtm_cubin_round_trip(ldtm_cubins):
    for cubin in ldtm_cubins.values():
        sass = run_cmd(["cuobjdump", "--dump-sass", cubin], cubin.parent)
        assert sorted(LOAD_WIDTHS.findall(sass)) == ["2", "4"]


@pytest.fixture(scope="module")
def sm100_context():
    try:
        cuda = ct.CDLL("libcuda.so.1")
    except OSError as error:
        pytest.skip(f"CUDA driver unavailable: {error}")

    def check(result):
        if result:
            reason = ct.c_char_p()
            cuda.cuGetErrorName(result, ct.byref(reason))
            pytest.fail(f"CUDA driver error {result}: {reason.value}")

    status = cuda.cuInit(0)
    if status == 100:
        pytest.skip("No visible CUDA device")
    check(status)
    count = ct.c_int()
    check(cuda.cuDeviceGetCount(ct.byref(count)))
    for ordinal in range(count.value):
        device, major, minor = ct.c_int(), ct.c_int(), ct.c_int()
        check(cuda.cuDeviceGet(ct.byref(device), ordinal))
        check(cuda.cuDeviceComputeCapability(ct.byref(major), ct.byref(minor), device))
        if (major.value, minor.value) == (10, 0):
            break
    else:
        pytest.skip("An SM100 GPU is required for the sm_100a cubin")

    context = ct.c_void_p()
    check(cuda.cuDevicePrimaryCtxRetain(ct.byref(context), device))
    try:
        check(cuda.cuCtxPushCurrent_v2(context))
        try:
            yield cuda, check
        finally:
            check(cuda.cuCtxPopCurrent_v2(ct.byref(context)))
    finally:
        check(cuda.cuDevicePrimaryCtxRelease_v2(device))


@pytest.mark.parametrize("variant", ["native", "lifted"])
def test_ldtm_lane_values(ldtm_cubins, sm100_context, variant):
    cuda, check = sm100_context
    module, kernel = ct.c_void_p(), ct.c_void_p()
    check(cuda.cuModuleLoad(ct.byref(module), os.fsencode(ldtm_cubins[variant])))
    try:
        check(cuda.cuModuleGetFunction(ct.byref(kernel), module, b"check_ldtm"))
        observed = (ct.c_uint32 * (32 * 6))()
        size = ct.c_size_t(ct.sizeof(observed))
        output = ct.c_uint64()
        check(cuda.cuMemAlloc_v2(ct.byref(output), size))
        try:
            check(cuda.cuMemsetD8_v2(output, 0xff, size))
            args = (ct.c_void_p * 1)(ct.addressof(output))
            check(cuda.cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, None, args, None))
            check(cuda.cuMemcpyDtoH_v2(observed, output, size))
        finally:
            check(cuda.cuMemFree_v2(output))
    finally:
        check(cuda.cuModuleUnload(module))

    for lane in range(32):
        expected = [4096 + 16 * lane + column for column in range(2, 8)]
        assert observed[6 * lane:6 * (lane + 1)] == expected, f"{variant}, lane {lane}"
