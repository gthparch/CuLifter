import ctypes
from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift.lifter_cpu import CpuLifter
from sir.instruction import Instruction
from sir.operand import Operand
from test.test_constant_bank import lift_const_bank_kernel


@pytest.mark.parametrize("width, source_type, result_type, value, predicate", [
    ("", ctypes.c_uint32, ctypes.c_uint32, 0x89ABCDEF, "P0"),
    (".U16", ctypes.c_uint16, ctypes.c_uint32, 0xABCD, "P0"),
    (".64", ctypes.c_uint64, ctypes.c_uint64, 0xFEDCBA9876543210, "P0"),
    ("", ctypes.c_uint32, ctypes.c_uint32, 0x1234ABCD, "!P0"),
])
def test_trailing_ldg_predicate_guards_memory_and_zero_fills(
    width, source_type, result_type, value, predicate
):
    store_width = ".64" if width == ".64" else ""
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        "MOV R4, c[0x0][0x168]", "MOV R5, c[0x0][0x16c]",
        "MOV R6, c[0x0][0x170]", "ISETP.NE.U32.AND P0, PT, R6, RZ, PT",
        f"LDG.E.LTC128B{width}.CONSTANT R8, [R4.64], {predicate}",
        f"STG.E{store_width} [R2.64], R8",
    ], 80, [(0x160, 8), (0x168, 8), (0x170, 4)])
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    compiled.triple = llvm.get_default_triple()
    machine = llvm.Target.from_default_triple().create_target_machine()
    compiled.data_layout = str(machine.target_data)
    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        signature = ctypes.CFUNCTYPE(None, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32)
        kernel = signature(engine.get_function_address("test_call"))
        source = source_type(value)
        output = result_type()
        enabled = int(predicate == "P0")
        kernel(ctypes.addressof(output), ctypes.addressof(source), enabled)
        assert output.value == value
        kernel(ctypes.addressof(output), ctypes.addressof(source), 1 - enabled)
        assert output.value == 0
        kernel(ctypes.addressof(output), 0, 1 - enabled)
        assert output.value == 0


def test_cpu_trailing_ldg_predicate_guards_memory():
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    module = ir.Module()
    module.triple = llvm.get_default_triple()
    module.data_layout = str(machine.target_data)
    word, address, flag = ir.IntType(32), ir.IntType(64), ir.IntType(1)
    function = ir.Function(module, ir.FunctionType(word, [address, flag]), "probe")
    builder = ir.IRBuilder(function.append_basic_block("entry"))
    operands = [Operand.parse(token) for token in ("R4", "[R2.64]", "P0")]
    inst = Instruction("0000", ["LDG", "E", "LTC128B", "CONSTANT"], operands)
    dest, = inst.get_defs()
    ptr, pred = inst.get_uses()
    dest.set_type_desc("Int32")
    ptr.set_type_desc("Int64")
    pred.set_type_desc("Bool")
    for part in ptr.reg_group:
        part.set_type_desc("Int32")
    registers = {
        "R2": builder.trunc(function.args[0], word),
        "R3": builder.trunc(builder.lshr(function.args[0], address(32)), word),
        "P0": function.args[1],
    }
    lifter = CpuLifter(SimpleNamespace(arg_map={}, const_data={}))
    lifter.llvm_module = module
    lifter.lift_instruction(inst, builder, registers, {})
    builder.ret(registers["R4"])
    compiled = llvm.parse_assembly(str(module))
    compiled.verify()

    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        signature = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint64, ctypes.c_bool)
        probe = signature(engine.get_function_address("probe"))
        source = ctypes.c_uint32(0x1234ABCD)
        assert probe(ctypes.addressof(source), True) == source.value
        assert probe(ctypes.addressof(source), False) == 0
        assert probe(0, False) == 0
