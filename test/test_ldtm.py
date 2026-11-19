from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift.lifter import UnsupportedInstructionException
from lift.lifter_nvvm import NVVMLifter
from sir.basicblock import BasicBlock
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import Operand
from transform.dce import DCE
from transform.typeanalysis import TypeAnalysis


@pytest.mark.parametrize("count", [1, 2, 4, 8, 16, 32, 64, 128])
def test_ldtm_registers_and_address(count):
    modifiers = [] if count == 1 else [f"x{count}"]
    inst = Instruction(0, ["LDTM", *modifiers], [Operand.parse("R4"), Operand.parse("tmem[UR6+0x60]")])
    expected = [f"R{4 + index}" for index in range(count)]
    assert [operand.reg for operand in inst.get_defs()] == expected
    address, = inst.get_uses()
    assert (address.reg, address.offset) == ("UR6", 0x60)
    kernel = Function("load")
    kernel.blocks = [BasicBlock("0", None, [inst])]
    TypeAnalysis().process_func(kernel)
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = ir.Module()
    word = ir.IntType(32)
    function = ir.Function(module, ir.FunctionType(word, [word]), "load")
    builder = ir.IRBuilder(function.append_basic_block("entry"))
    registers = {"UR6": function.args[0]}
    lifter.lift_instruction(inst, builder, registers, {})
    builder.ret(registers[expected[-1]])
    llvm.parse_assembly(str(module)).verify()
    call, = [instruction for instruction in builder.block.instructions if instruction.opname == "call"]
    addr, = call.args
    assert addr.opname == "add" and addr.operands[0] is function.args[0] and addr.operands[1].constant == 96
    assert f"tcgen05.ld.sync.aligned.32x32b.x{count}.b32" in call.callee.asm
    assert "tcgen05.wait::ld.sync.aligned;" in call.callee.asm
    if count == 1:
        assert registers["R4"] is call
    else:
        for index, register in enumerate(expected):
            assert registers[register].operands == [call] and registers[register].indices == [index]


def test_other_ldtm_layout_is_reported():
    inst = Instruction(0, ["LDTM", "16dp64bit", "x2"], [Operand.parse("R4"), Operand.parse("tmem[UR6]")])
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    with pytest.raises(UnsupportedInstructionException, match="Unsupported LDTM form"):
        lifter.lift_instruction(inst, None, {}, {})


def test_tmem_side_effects_survive_dce():
    instructions = [
        Instruction(0, ["UTCIMMA"], [Operand.parse(token) for token in
                    ("gdesc[UR32]", "gdesc[UR30]", "tmem[UR26]", "tmem[UR16]", "idesc[UR17]", "UP2")]),
        Instruction(16, ["UTCBAR"], [Operand.parse("[UR18]"), Operand.parse("URZ")]),
        Instruction(32, ["LDTM", "x4"], [Operand.parse("R108"), Operand.parse("tmem[UR12]")]),
        Instruction(48, ["UTCATOMSWS", "AND"], [Operand.parse("URZ"), Operand.parse("UR4")]),
    ]
    kernel = Function("tensor_memory")
    kernel.blocks = [BasicBlock("0", None, instructions)]
    DCE().process(kernel)
    assert kernel.blocks[0].instructions == instructions
