import ctypes
from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift import lifter_cpu
from lift.lifter_nvvm import NVVMLifter
from sir.instruction import Instruction
from sir.operand import Operand


@pytest.mark.parametrize("lifter_type, address_space", [(lifter_cpu.CpuLifter, 0), (NVVMLifter, 3)])
@pytest.mark.parametrize("opcode, expected", [
    ("ATOMS.MIN", 1),
    ("ATOMS.MIN.S32", 0x80000000),
    ("ATOMS.MAX", 0x80000000),
    ("ATOMS.MAX.S32", 1),
])
def test_shared_atomic_minmax_uses_opcode_signedness(monkeypatch, lifter_type, address_space, opcode, expected):
    monkeypatch.setattr(lifter_cpu, "_LOCKSTEP", False)
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()

    lifter = lifter_type(SimpleNamespace(arg_map={}, const_data={}))
    module = ir.Module()
    module.triple = llvm.get_default_triple()
    module.data_layout = str(machine.target_data)

    i32 = ir.IntType(32)
    memory_type = ir.ArrayType(ir.IntType(8), 4)
    shared = ir.GlobalVariable(module, memory_type, "shared_mem", addrspace=address_space)
    shared.initializer = ir.Constant(memory_type, None)
    shared.align = 4
    lifter.SharedMem = shared

    function = ir.Function(module, ir.FunctionType(i32, [i32, i32]), "probe")
    builder = ir.IRBuilder(function.append_basic_block("entry"))
    address = builder.bitcast(shared, ir.PointerType(i32, address_space))
    builder.store(function.args[0], address)

    operands = [Operand.parse(text) for text in ("RZ", "[R0]", "R1")]
    for operand in operands:
        operand.type_desc = "UInt32"
    instruction = Instruction("0000", opcode.split("."), operands)
    registers = {"R0": ir.Constant(i32, 0), "R1": function.args[1]}
    lifter.lift_instruction(instruction, builder, registers, {})
    builder.ret(builder.load(address))

    compiled = llvm.parse_assembly(str(module))
    compiled.verify()
    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        signature = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
        probe = signature(engine.get_function_address("probe"))
        assert probe(1, 0x80000000) == expected
