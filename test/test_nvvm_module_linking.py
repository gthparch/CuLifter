import ctypes
import io
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from llvmlite import binding as llvm

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "test" / "addrtest"
COUNTER = FIXTURES / "CB4b_device_int_rw" / "CB4b_device_int_rw.cubin"
MULTIPLIER = FIXTURES / "CB4k_runtime_update" / "CB4k_runtime_update.cubin"


def lift_kernel(name, image, lines):
    elf = Elf(image)
    (original,) = elf.arg_map
    elf.arg_map[name] = [(0x210, 8)]
    elf.const_data[name] = elf.const_data.get(original, {})

    sass = ["code for sm_90", f"Function : {name}"]
    sass.extend(
        f"/*{index * 16:04x}*/ {line} ; /* 0x0000000000000000 */"
        for index, line in enumerate([*lines, "EXIT"])
    )
    module = Module(name, SaSSParser("\n".join(sass), elf=elf))
    lifter = NVVMLifter(elf, linkable=True)
    out = io.StringIO()
    lifter.lift_module(module, out)
    assert not lifter.placeholder_log
    ir = llvm.parse_assembly(out.getvalue())
    ir.verify()
    (const_mem,) = (g for g in ir.global_variables if g.name == "const_mem")
    assert const_mem.linkage == llvm.Linkage.internal
    return ir


def test_separately_lifted_kernels_share_only_their_own_cubin_state(tmp_path):
    same_image = shutil.copyfile(COUNTER, tmp_path / "copied.cubin")
    writer = lift_kernel("set_counter", COUNTER, [
        "ULDC.64 UR4, c[0x4][0x0]",
        "MOV R6, 0x2a",
        "STG.E [UR4.64], R6",
    ])
    read = [
        "ULDC.64 UR4, c[0x4][0x0]",
        "LDG.E R6, [UR4.64]",
        "LDC.64 R2, c[0x0][0x210]",
        "STG.E [R2.64], R6",
    ]
    writer.link_in(lift_kernel("read_counter", same_image, read))
    writer.link_in(lift_kernel("read_unrelated_global", MULTIPLIER, read))
    writer.verify()
    globals_in_linked_module = [
        g for g in writer.global_variables if g.name.startswith("__slifter_nv_global_")
    ]
    assert len(globals_in_linked_module) == 2
    assert all(g.linkage == llvm.Linkage.linkonce_odr for g in globals_in_linked_module)

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    writer.triple = llvm.get_default_triple()
    machine = llvm.Target.from_default_triple().create_target_machine()
    writer.data_layout = str(machine.target_data)
    with llvm.create_mcjit_compiler(writer, machine) as engine:
        engine.finalize_object()
        kernel = ctypes.CFUNCTYPE(None, ctypes.c_uint64)
        output = ctypes.c_int32(-1)
        kernel(engine.get_function_address("set_counter"))(0)
        kernel(engine.get_function_address("read_counter"))(ctypes.addressof(output))
        assert output.value == 42
        kernel(engine.get_function_address("read_unrelated_global"))(ctypes.addressof(output))
        assert output.value == 0


@pytest.mark.parametrize("image, sass, global_names", [
    (COUNTER, COUNTER.with_name(COUNTER.stem + "_sm90.sass"), {"__slifter_nv_global"}),
    (
        ROOT / "test" / "sm90_synth" / "nvglobal_table" / "kernel.cubin",
        ROOT / "test" / "sm90_synth" / "nvglobal_table" / "kernel.sass",
        {"__slifter_nv_global", "__slifter_nv_global_init"},
    ),
])
def test_cli_keeps_legacy_symbols_unless_linking_is_requested(tmp_path, image, sass, global_names):
    for options in ([], ["--linkable-nvvm"]):
        dest = tmp_path / "counter"
        subprocess.run(
            [sys.executable, str(ROOT / "main.py"), "-i", str(sass), str(image),
             "-o", str(dest), "--lifter", "nvvm", *options],
            check=True, capture_output=True, text=True,
        )
        ir = llvm.parse_assembly(dest.with_suffix(".ll").read_text())
        ir.verify()
        device_globals = [g for g in ir.global_variables if g.name.startswith("__slifter_nv_global")]
        if options:
            assert {g.name.rsplit("_", 1)[0] for g in device_globals} == global_names
            assert all(g.linkage == llvm.Linkage.linkonce_odr for g in device_globals)
        else:
            assert {g.name for g in device_globals} == global_names
            assert all(g.linkage == llvm.Linkage.external for g in device_globals)
