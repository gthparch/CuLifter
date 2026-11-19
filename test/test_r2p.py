from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.basicblock import BasicBlock
from transform.typeanalysis import TypeAnalysis


@pytest.mark.parametrize(
    "operands, encoding, mask, bits, byte_offset",
    [
        ("R99", "000000ff63007804", 0xff, range(7), 0),
        ("R99.B1, 0x21", "0000002163007804", 0x21, (0, 5), 8),
    ],
)
def test_r2p_mask_selects_predicate_defs(operands, encoding, mask, bits, byte_offset):
    text = f"""code for sm_80
Function : probe
/*0000*/ R2P PR, {operands}; /* 0x{encoding} */
"""
    func, = SaSSParser(text, None).apply()
    inst, = func.instructions
    predicates = [f"P{bit}" for bit in bits]
    assert [operand.reg for operand in inst.get_defs()] == predicates
    source, mask_operand = inst.get_uses()
    assert source.reg == "R99" and mask_operand.immediate_value == mask

    func.blocks = [BasicBlock("0", None, [inst])]
    TypeAnalysis().process_func(func)
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = ir.Module()
    word = ir.IntType(32)
    function_type = ir.FunctionType(word, [word])
    function = ir.Function(module, function_type, "probe")
    builder = ir.IRBuilder(function.append_basic_block("entry"))
    registers = {"R99": function.args[0]}
    lifter.lift_instruction(inst, builder, registers, {})
    builder.ret(registers[predicates[-1]])
    llvm.parse_assembly(str(module)).verify()

    assert set(registers) == {"R99", *predicates}
    for bit, predicate in zip(bits, predicates):
        truncated, = registers[predicate].operands
        shifted, = truncated.operands
        assert shifted.opname == "lshr"
        assert shifted.operands[1].constant == bit + byte_offset
