"""Blackwell UIADD3.64 with discarded carries and a zero addend uses pairs."""
import ctypes
from types import SimpleNamespace

from llvmlite import binding as llvm, ir
import pytest

from lift.lifter_nvvm import NVVMLifter
from sir.instruction import Instruction
from sir.operand import Operand


def parse(args):
    return Instruction("0000", ["UIADD3", "64"], [Operand.parse(s) for s in args])


@pytest.mark.parametrize("predicates", [[], ["UPT", "UPT"]])
@pytest.mark.parametrize("subtract", [False, True])
def test_pair_carry_and_borrow(predicates, subtract):
    # Use the same register as source and destination, as in the real binary.
    operands = ["UR8", *predicates, "UR18", "-UR8" if subtract else "UR8", "URZ"]
    instruction = parse(operands)
    assert instruction.opcodes == ["IADD64"]
    assert len(instruction.get_defs()) == 1
    assert [tuple(x.reg for x in op.reg_group) for op in instruction.operands] == [
        ("UR8", "UR9"), ("UR18", "UR19"), ("UR8", "UR9")
    ]
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = lifter.llvm_module = ir.Module()
    module.triple = llvm.get_default_triple()
    module.data_layout = str(machine.target_data)
    i32, i64 = ir.IntType(32), ir.IntType(64)
    fn = ir.Function(module, ir.FunctionType(i64, [i64, i64]), "probe")
    b = ir.IRBuilder(fn.append_basic_block("entry"))
    regs = {}
    for key, val in zip((18, 8), fn.args):
        regs[f"UR{key}"] = b.trunc(val, i32)
        regs[f"UR{key + 1}"] = b.trunc(b.lshr(val, ir.Constant(i64,32)), i32)
    for operand in instruction.operands:
        operand.type_desc = "Int64"
    lifter.lift_instruction(instruction,b,regs,{})
    high = b.shl(b.zext(regs["UR9"],i64),ir.Constant(i64,32))
    b.ret(b.or_(high,b.zext(regs["UR8"],i64)))
    compiled = llvm.parse_assembly(str(module))
    compiled.verify()
    with llvm.create_mcjit_compiler(compiled,machine) as engine:
        engine.finalize_object()
        probe = ctypes.CFUNCTYPE(ctypes.c_uint64,ctypes.c_uint64,ctypes.c_uint64)(engine.get_function_address("probe"))
        for a, c in [(0xffffffff,1),(1,0xffffffff),(0x100000000,1),(0x100000001,0x200000001),(0xffffffffffffffff,1)]:
            expected = a-c if subtract else a+c
            assert probe(a,c) == expected & 0xffffffffffffffff


@pytest.mark.parametrize("args", [
    ["UR8", "UP0", "UPT", "UR18", "UR8", "URZ"],
    ["UR8", "UPT", "UPT", "UR18", "UR8", "UR10"],
])
def test_live_carry_or_third_source_is_not_discarded(args):
    assert parse(args).opcodes == ["UIADD3","64"]
