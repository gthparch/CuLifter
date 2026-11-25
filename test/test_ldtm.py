from pathlib import Path
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


@pytest.mark.parametrize("shape,modifier,words,count", [
    (shape, modifier, words, count)
    for shape, modifier, words in (("32x32b", None, 1), ("16x64b", "16dp64bit", 1),
                                   ("16x128b", "16dp128bit", 2), ("16x256b", "16dp256bit", 4))
    for count in (1, 2, 4, 8, 16, 32, 64, 128) if count >= words
])
def test_ldtm_registers_and_address(shape, modifier, words, count):
    repeat = count // words
    modifiers = ([modifier] if modifier else []) + ([] if repeat == 1 else [f"x{repeat}"])
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
    assert f"tcgen05.ld.sync.aligned.{shape}.x{repeat}.b32" in call.callee.asm
    assert "tcgen05.wait::ld.sync.aligned;" in call.callee.asm
    if count == 1:
        assert registers["R4"] is call
    else:
        for index, register in enumerate(expected):
            assert registers[register].operands == [call] and registers[register].indices == [index]


def test_other_ldtm_layout_is_reported():
    inst = Instruction(0, ["LDTM", "16dp32bit_t0_t15", "x2"], [Operand.parse("R4"), Operand.parse("tmem[UR6]")])
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

@pytest.mark.parametrize("old,new,recognized", [
    ("", "", True),
    ("/*0270*/ UMOV UR4, 0x400", "/*0270*/ UMOV UR4, 0x404", False),
    ("/*0230*/ S2UR UR5, SR_CgaCtaId", "/*0230*/ S2UR UR5, SR_CgaCtaSize", False),
    ("/*0050*/ @P1 BRA 0x220", "/*0050*/ @P1 BRA 0x350", False),
    ("/*0350*/ ULEA UR4, UR5, UR4, 0x18", "/*0350*/ @P0 ULEA UR4, UR5, UR4, 0x18", False),
])
def test_ranked_slot_allocation(old, new, recognized):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    source = (Path(__file__).parent / "sm100_optest/column_alloc_ranked.sass").read_text()
    if old:
        assert old in source
        source = source.replace(old, new)
    func, = Module("transfer", SaSSParser(source, None)).functions
    recover_single_aligned(func)
    alloc = [inst for inst in func.instructions if inst.opcodes == ["TMEMALLOC1"]]
    assert len(alloc) == int(recognized)
    if recognized:
        assert [op.immediate_value for op in alloc[0].operands] == [0x400, 128]

@pytest.mark.parametrize("entry,upper,fold", [
    ("NOP", "LDTM.16dp32bit_t16_t31.x2 R4, tmem[UR6+0x100004]", True),
    ("@P0 BRA 0x20", "LDTM.16dp32bit_t16_t31.x2 R4, tmem[UR6+0x100004]", False),
    ("NOP", "LDTM.16dp32bit_t16_t31.x2 R6, tmem[UR6+0x100004]", False),
    ("NOP", "LDTM.16dp32bit_t16_t31.x2 R4, tmem[UR7+0x100004]", False),
])
def test_split_transfer_provenance(entry, upper, fold):
    from .test_tcgen05 import parse_kernel
    from transform.tmem_column import recover_split_transfers

    func, = parse_kernel([entry, "LDTM.16dp32bit_t0_t15.x2 R4, tmem[UR6]", upper],
                         100, "test_call", None).functions
    recover_split_transfers(func)
    first, last = func.instructions[1:3]
    assert (last.opcodes == ["NOP"]) == fold
    if fold:
        assert first.opcodes == ["LDTM", "SPLIT", "x2"]
        assert first.get_uses()[-1].immediate_value == 0x100004

@pytest.mark.parametrize("old,new,static_bytes,fold", [
    ("", "", 0x404, True),
    ("", "", 0, False),       # no ELF proof of both reserved symbols and CUDA entry
    ("", "", 0x400, False),   # publication slot lies outside modeled shared memory
    ("@P0 BRA 0x2f0", "@!P0 BRA 0x2f0", 0x404, False),
    ("@!P0 BRA 0x1c0", "@P0 BRA 0x1c0", 0x404, False),
    ("/*03e0*/ UMOV UR4, 0x400", "/*03e0*/ UMOV UR4, 0x404", 0x404, False),
    ("/*1050*/ MOV R2, 0x1070", "/*1050*/ MOV R2, 0x10a0", 0x404, False),
    ("/*1110*/ RET.REL.NODEC R4 0x0", "/*1110*/ RET.REL.NODEC R2 0x0", 0x404, False),
])
def test_129_guarded_slot_preserves_permit_and_release(old, new, static_bytes, fold):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    source = (Path(__file__).parent / "sm100_optest/column_alloc_guarded_129.sass").read_text()
    if old:
        assert old in source
        source = source.replace(old, new)
    func, = Module("transfer", SaSSParser(source, None)).functions
    recover_single_aligned(func, guarded_static_bytes=static_bytes)
    sites = [i for i, inst in enumerate(func.instructions) if inst.opcodes == ["TMEMALLOC1"]]
    assert len(sites) == int(fold)
    if fold:
        i, = sites
        assert func.instructions[i-1].opcodes == ["BRA"] and func.instructions[i-1].pflag
        assert func.instructions[i+1].opcodes == ["ELECT"]
        assert sum(inst.is_call() for inst in func.instructions) == 3
        assert sum(inst.opcodes[-1] == "ALIGNED_RELEASE" for inst in func.instructions) == 1
