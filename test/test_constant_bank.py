import ctypes
import io
import re
from pathlib import Path

import pytest
from llvmlite import binding as llvm

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.module import Module
from utils.elf import Elf


def lift_const_bank_kernel(lines, sm, args, *, stack_size=0):
    elf = Elf(Path(__file__).parent / "sm100_optest" / "test_call_sm90.elf")
    name = "test_call"
    elf.arg_map[name] = args
    elf.frame_size[name] = stack_size
    elf.min_stack_size[name] = stack_size
    sass = [f"code for sm_{sm}", f"Function : {name}"]
    for index, line in enumerate([*lines, "EXIT"]):
        sass.append(f"/*{16 * index:04x}*/ {line} ; /* 0x0000000000000000 */")
    module = Module("test", SaSSParser("\n".join(sass), elf))
    out = io.StringIO()
    lifter = NVVMLifter(elf)
    lifter.lift_module(module, out)
    assert not lifter.placeholder_log
    compiled = llvm.parse_assembly(out.getvalue())
    compiled.verify()
    return compiled


def run_with_param_bits(compiled, result, bits):
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    compiled.triple = llvm.get_default_triple()
    machine = llvm.Target.from_default_triple().create_target_machine()
    compiled.data_layout = str(machine.target_data)
    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        signature = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64)
        kernel = signature(engine.get_function_address("test_call"))
        kernel(ctypes.addressof(result), bits)


@pytest.mark.parametrize("bits,expected", [
    (0x3FE0000000000000, 0.5),
    (0xBFF199999999999A, ctypes.c_float(-1.1).value),
])
def test_f2f_reads_double_from_constant_bank(bits, expected):
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        "F2F.F32.F64 R4, c[0x0][0x198]",
        "STG.E [R2.64], R4",
    ], 80, [(0x160, 8), (0x198, 8)])
    result = ctypes.c_float()
    run_with_param_bits(compiled, result, bits)
    assert result.value == expected


@pytest.mark.parametrize("sm", [70, 80])
@pytest.mark.parametrize("prologue", [
    "MOV R1, c[0x0][0x28]",
    "IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28]",
])
def test_stack_base_is_aligned_and_independent_of_ctaid_z(sm, prologue):
    compiled = lift_const_bank_kernel([
        prologue, "IADD3 R1, R1, -0x68, RZ",
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        "STL [R1], R2", "LDL R4, [R1]", "STG.E [R2.64], R4",
        "S2R R8, SR_CTAID.Z", "STG.E [R2.64+0x4], R8",
        "STG.E [R2.64+0x8], R1",
    ], sm, [(0x160, 8)], stack_size=0x68)
    llvm.initialize_all_targets()
    machine = llvm.Target.from_triple(compiled.triple).create_target_machine(cpu="sm_80")
    passes = llvm.create_new_module_pass_manager()
    passes.add_instruction_combine_pass()
    passes.run(compiled, llvm.create_pass_builder(machine, llvm.PipelineTuningOptions()))
    ir = str(compiled.get_function("test_call"))
    assert len(re.findall(r"call i32 @llvm\.nvvm\.read\.ptx\.sreg\.ctaid\.z\(", ir)) == 1
    assert "store i32 8, ptr addrspace(1)" in ir


def test_ampere_uldc64_reads_parameter_value_at_0x198():
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        "ULDC.64 UR4, c[0x0][0x198]",
        "STG.E [R2.64], UR4", "STG.E [R2.64+0x4], UR5",
    ], 80, [(0x160, 8), (0x198, 8)])
    result = ctypes.c_uint64()
    run_with_param_bits(compiled, result, 0xFEDCBA9876543210)
    assert result.value == 0xFEDCBA9876543210


@pytest.mark.parametrize("first_offset", [0, 0x30, 0x70])
def test_hopper_uldc64_materializes_parameter_address_at_0x198(first_offset):
    arg0 = 0x210 + first_offset
    compiled = lift_const_bank_kernel([
        f"LDC.64 R2, c[0x0][{arg0:#x}]",
        "ULDC.64 UR4, c[0x0][0x198]",
        "STG.E [R2.64], UR4", "STG.E [R2.64+0x4], UR5",
    ], 90, [(arg0, 8)])
    ir = str(compiled.get_function("test_call"))
    assert "cvta.param.u64" in ir
    assert re.search(rf"sub i64 .*?, {first_offset}\b", ir)
