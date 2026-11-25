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
@pytest.mark.parametrize("packed", [False, True])
def test_ldtm_registers_and_address(shape, modifier, words, count, packed):
    repeat = count // words
    modifiers = ([modifier] if modifier else []) + ([] if repeat == 1 else [f"x{repeat}"])
    if packed:
        modifiers.append("PACK16BIT")
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
    qualifier = ".pack::16b" if packed else ""
    assert f"tcgen05.ld.sync.aligned.{shape}.x{repeat}{qualifier}.b32" in call.callee.asm
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


@pytest.mark.parametrize("opcodes", [
    ["LDTM", "EXPAND16BIT"],
    ["STTM", "PACK16BIT"],
    ["LDTM", "PACK16BIT", "x2"],
    ["STTM", "16dp64bit", "x2", "EXPAND16BIT", "EXPAND16BIT"],
])
def test_invalid_tmem_packing_modifiers(opcodes):
    assert Instruction(0, opcodes, []).tmem_transfer is None


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


@pytest.mark.parametrize("old,new,fold", [
    ("", "", True),
    ("DEPBAR.LE SB0", "DEPBAR.LE SB1", True),
    ("/*0420*/ DEPBAR.LE SB0", "/*0420*/ DEPBAR.LE SB1", False),
    ("/*05c0*/ IMAD.MOV.U32 R2, RZ, RZ, -0x1", "/*05c0*/ MOV R2, 0xffffffff", True),
    ("/*0060*/ LDC.64 R2, c[0x0][0x380]", "/*0060*/ @P1 BRA 0x80", False),
    ("/*0020*/ S2UR UR8, SR_CgaCtaId", "/*0020*/ S2UR UR8, SR_TID.X", False),
    ("/*0670*/ UMOV UR8, 0x400", "/*0670*/ UMOV UR8, 0x404", False),
    ("ULEA UR13, UR9, UR8, 0x18", "ULEA UR13, -UR9, UR8, 0x18", False),
    ("ULEA UR13, UR9, UR8, 0x18", "ULEA UR13, UR9, -UR8, 0x18", False),
    ("ULEA UR13, UR9, UR8, 0x18", "ULEA UR13, UR9.reuse, UR8, 0x18", True),
    ("/*0b60*/ LDS R0, [UR13+0x2408]", "/*0b60*/ LDS R0, [UR13+0x240c]", False),
    ("/*0de0*/ SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [UR8], RZ",
     "/*0de0*/ UMOV UR13, 0x2808", False),
    ("@!P1 BRA 0x210", "@!P1 BRA 0x400", False),
    ("BSSY.RECONVERGENT B0, 0x170", "BSSY.RECONVERGENT B0, 0x400", False),
    ("/*0d80*/ CALL.REL.NOINC 0xe10", "/*0d80*/ CALL.REL.NOINC 0xe40", False),
])
def test_129_guarded_unit_checks_ranked_slot_across_mma(old, new, fold):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    source = (Path(__file__).parent / "sm100_optest/column_alloc_guarded_unit_129.sass").read_text()
    if old:
        assert old in source
        source = source.replace(old, new)
    func, = Module("mma_case", SaSSParser(source, None)).functions
    recover_single_aligned(func, guarded_static_bytes=0x280c)
    sites = [inst for inst in func.instructions if inst.opcodes == ["TMEMALLOC1"]]
    assert len(sites) == int(fold)
    if fold:
        assert [op.immediate_value for op in sites[0].operands] == [0x2808, 32]
        assert sum(inst.opcodes[-1] == "ALIGNED_RELEASE" for inst in func.instructions) == 1
        assert sum(inst.is_call() for inst in func.instructions) == 3


@pytest.mark.parametrize("schedule,slot,columns,old,new", [
    ("local_slot", 0x2808, 512, "", ""),
    ("local_slot", 0x2808, 512, "/*00a0*/ UMOV UR7, 0x400", "/*00a0*/ UMOV UR7, 0x404"),
    ("local_slot", 0x2808, 512, "/*0090*/ S2UR UR8, SR_CgaCtaId", "/*0090*/ S2UR UR8, SR_TID.X"),
    ("local_slot", 0x2808, 512, "/*05e0*/ ULEA UR4, UR8, UR7, 0x18", "/*05e0*/ ULEA UR4, UR8, UR6, 0x18"),
    ("local_slot", 0x2808, 512, "/*0360*/ @P0 BRA 0x650", "/*0360*/ @P0 BRA 0x590"),
    ("multi", 0x2400, 128, "", ""),
    ("multi", 0x2400, 128, "/*0670*/ UMOV UR10, 0x400", "/*0670*/ UMOV UR10, 0x404"),
    ("multi", 0x2400, 128, "/*03f0*/ IMAD.MOV.U32 R5, RZ, RZ, 0xf", "/*03f0*/ IMAD.MOV.U32 R5, RZ, RZ, 0x7"),
    ("multi", 0x2400, 128, "/*0f70*/ BSYNC.RECONVERGENT B0", "/*0f70*/ BSYNC B0"),
    ("multi", 0x2400, 128, "/*0600*/ WARPSYNC.ALL", "/*0600*/ BMOV B0, R4"),
    ("unit_gpr", 0xc08, 32, "", ""),
    ("unit_gpr", 0xc08, 32, "/*0350*/ IMAD.MOV.U32 R7, RZ, RZ, 0x1", "/*0350*/ IMAD.MOV.U32 R7, RZ, RZ, 0x3"),
    ("unit_gpr", 0xc08, 32, "/*0530*/ STS [R9+0x808]", "/*0530*/ STS [R9+0x80c]"),
    ("multi_gpr", 0x808, 64, "", ""),
    ("multi_gpr", 0x808, 64, "/*02d0*/ MOV R6, 0x400", "/*02d0*/ MOV R6, 0x404"),
    ("multi_gpr", 0x808, 64, "STS [R9+0x408], R2", "STS [R9+0x40c], R2"),
    ("multi_gpr", 0x808, 64, "/*0310*/ @P0 BRA 0x520", "/*0310*/ @P0 BRA 0x530"),
])
def test_129_guarded_multi_column_protocol(schedule, slot, columns, old, new):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    fixture = Path(__file__).parent / "sm100_optest" / f"column_alloc_guarded_{schedule}_129.sass"
    source = fixture.read_text()
    if old:
        assert source.count(old) == 1
        source = source.replace(old, new)
    func, = Module("column", SaSSParser(source, None)).functions
    recover_single_aligned(func, guarded_static_bytes=slot + 4)
    alloc = [i for i in func.instructions if i.opcodes == ["TMEMALLOC1"]]
    assert len(alloc) == int(not old)
    assert sum(i.opcodes[-1] == "ALIGNED_RELEASE" for i in func.instructions) == len(alloc)
    if alloc:
        assert [op.immediate_value for op in alloc[0].operands] == [slot, columns]
        assert sum(i.is_call() for i in func.instructions) == 3

@pytest.mark.parametrize("consume_result", [True, False])
def test_129_release_reuses_dead_slot_base(consume_result):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    fixture = Path(__file__).parent / "sm100_optest/column_alloc_guarded_multi_129.sass"
    prefix, release = fixture.read_text().split("REDUX UR6, R2", 1)
    if consume_result:
        release = release.replace("RZ, RZ, UR6", "RZ, RZ, UR16", 1)
    func, = Module("column", SaSSParser(prefix + "REDUX UR16, R2" + release, None)).functions
    recover_single_aligned(func, guarded_static_bytes=0x2404)
    assert sum(i.opcodes == ["TMEMALLOC1"] for i in func.instructions) == int(consume_result)

@pytest.mark.parametrize("mantissa,recognized", [("5.9604644775390625e-08", True), ("0", False)])
def test_13_reordered_tmem_ownership_literals(mantissa, recognized):
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    source = (Path(__file__).parent / "sm100_optest/column_alloc_128.sass").read_text()
    source = source.replace("/*0390*/ IMAD.MOV.U32 R5, RZ, RZ, 0xf",
                            f"/*0390*/ HFMA2 R7, -RZ, RZ, 0, {mantissa}")
    source = source.replace("/*03b0*/ IMAD.MOV.U32 R7, RZ, RZ, 0x1",
                            "/*03b0*/ IMAD.MOV.U32 R5, RZ, RZ, 0xf")
    func, = Module("column", SaSSParser(source, None)).functions
    recover_single_aligned(func)
    assert sum(inst.opcodes == ["TMEMALLOC1"] for inst in func.instructions) == int(recognized)
