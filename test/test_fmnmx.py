import ctypes
from types import SimpleNamespace

from llvmlite import binding as llvm, ir
import pytest

from lift.lifter_nvvm import NVVMLifter
from sir.instruction import Instruction
from sir.operand import Operand


@pytest.mark.parametrize("predicate, expected_signed_zero", [("PT", 0x80000000), ("!PT", 0)])
def test_plain_fmnmx_obeys_numeric_operand_and_signed_zeros(predicate, expected_signed_zero):
    """In PTX, plain min/max returns the number, even when the other source is NaN."""
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = lifter.llvm_module = ir.Module()
    module.triple = llvm.get_default_triple()
    module.data_layout = str(machine.target_data)

    i32, f32 = ir.IntType(32), ir.FloatType()
    fn = ir.Function(module, ir.FunctionType(i32, [i32,i32]), "probe")
    builder = ir.IRBuilder(fn.append_basic_block("entry"))
    registers = {key:builder.bitcast(value,f32) for key,value in zip(("R1","R2"),fn.args)}
    operands = [Operand.parse(text) for text in ("R0","R1","R2",predicate)]
    for operand in operands[:3]:
        operand.type_desc = "Float32"
    instruction = Instruction("0000", ["FMNMX"], operands)
    lifter.lift_instruction(instruction,builder,registers,{})
    builder.ret(builder.bitcast(registers["R0"],i32))

    compiled = llvm.parse_assembly(str(module))
    compiled.verify()
    with llvm.create_mcjit_compiler(compiled,machine) as engine:
        engine.finalize_object()
        probe = ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32)(engine.get_function_address("probe"))
        numeric, qnan = 0xc0000000, 0x7fc00000
        assert probe(numeric,qnan)==numeric
        assert probe(qnan,numeric)==numeric
        assert probe(0x7f800001,numeric)==numeric
        assert probe(numeric,0x7f800001)==numeric
        assert probe(0,0x80000000)==expected_signed_zero
        assert probe(0x80000000,0)==expected_signed_zero
        assert probe(0x80000000,0x80000000)==0x80000000
        assert probe(0,0)==0
        assert probe(1,2)==(1 if predicate=="PT" else 2)
