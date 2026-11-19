import ctypes
import struct
from itertools import product
from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift.lifter_cpu import CpuLifter
from lift.lifter_nvvm import NVVMLifter
from parse.parser_base import SaSSParserBase
from sir.basicblock import BasicBlock
from sir.function import Function
from transform.typeanalysis import TypeAnalysis


def parse_sass(line):
    parser = SaSSParserBase(None, "", None)
    opcode, guard, operands = parser.get_inst_opcode(line)
    return parser.parse_instruction("0000", opcode, guard, parser.get_inst_operands(operands),
                                    operands, None)


def check_registers(lines, inputs, outputs, cases, backend=NVVMLifter):
    """Execute scalar lifted IR on the host and compare raw 32-bit register words."""
    lifter = backend(SimpleNamespace(arg_map={}, const_data={}))
    module = lifter.llvm_module = ir.Module()
    i32 = ir.IntType(32)
    signature = ir.FunctionType(ir.VoidType(), [i32] * len(inputs) + [i32.as_pointer()])
    func = ir.Function(module, signature, "probe")
    builder = ir.IRBuilder(func.append_basic_block("entry"))
    regs = dict(zip(inputs, func.args))
    block = BasicBlock("0", None, [parse_sass(line) for line in lines])
    kernel = Function("probe")
    kernel.blocks = [block]
    TypeAnalysis().process_func(kernel)
    for inst in block.instructions:
        lifter.lift_instruction(inst, builder, regs, {})
    for index, reg in enumerate(outputs):
        word = builder.bitcast(regs[reg], i32)
        builder.store(word, builder.gep(func.args[-1], [ir.Constant(i32, index)]))
    builder.ret_void()

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    target = llvm.Target.from_default_triple().create_target_machine()
    compiled = llvm.parse_assembly(str(module))
    compiled.verify()
    with llvm.create_mcjit_compiler(compiled, target) as engine:
        engine.finalize_object()
        u32 = ctypes.c_uint32
        call = ctypes.CFUNCTYPE(None, *([u32] * len(inputs)), ctypes.POINTER(u32))(
            engine.get_function_address("probe"))
        for args, expected in cases:
            result = (u32 * len(outputs))(*([0xdeadbeef] * len(outputs)))
            call(*args, result)
            assert list(result) == list(expected), (lines, args)


def float_bits(value):
    return int.from_bytes(struct.pack("<f", value), "little")


def test_ffma2_swizzle_negation_and_overlapping_accumulator():
    lines = [
        "FFMA2 R10, R9.F32, R4.F32x2.HI_LO, R6.F32x2.HI_LO",
        "FFMA2 R12, -R4.reuse.F32x2.LO_HI.NP, R9.reuse.F32, R6.reuse.F32x2.HI_LO",
        "FFMA2 R6, R4.F32x2.HI_LO.NP, R9.F32, R6.F32x2.HI_LO",
    ]
    args = [float_bits(v) for v in (2, 3, 5, 7, 11)]
    expected = [float_bits(v) for v in (27, 40, -28, 29, 27, -26)]
    check_registers(lines, ["R4", "R5", "R6", "R7", "R9"],
                    ["R10", "R11", "R12", "R13", "R6", "R7"], [(args, expected)])


def test_ffma2_rounds_each_lane_only_after_the_addition():
    args = [float_bits(v) for v in (1 + 2**-23, 1 - 2**-23, 1 - 2**-23, -1 + 2**-23, -1, 1)]
    expected = [float_bits(-(2**-46)), float_bits(2**-22 - 2**-46)]
    check_registers(["FFMA2 R8, R0.F32x2.HI_LO, R2.F32x2.HI_LO, R4.F32x2.HI_LO"],
                    ["R0", "R1", "R2", "R3", "R4", "R5"], ["R8", "R9"], [(args, expected)])


@pytest.mark.parametrize("opcode,prefix", [("PLOP3", "P"), ("UPLOP3", "UP")])
@pytest.mark.parametrize("discard", [None, 0, 1])
def test_two_predicate_truth_tables(opcode, prefix, discard):
    outputs = [prefix + "3", prefix + "4"]
    dest = [prefix + "T" if i == discard else reg for i, reg in enumerate(outputs)]
    inputs = [prefix + str(i) for i in range(3)]
    line = f"{opcode}.LUT " + ", ".join([*dest, *inputs, "0xf8", "0x8f"])
    cases = []
    for index, bits in enumerate(product((0, 1), repeat=3)):
        expected = [(table >> index) & 1 for table in (0xf8, 0x8f)]
        cases.append((bits, [v for i, v in enumerate(expected) if i != discard]))
    check_registers([line], inputs, [reg for i, reg in enumerate(outputs) if i != discard], cases)


@pytest.mark.parametrize("byte", range(4))
def test_p2r_replaces_only_masked_bits_in_selected_byte(byte):
    line = f"P2R.B{byte} R1, PR, R0, 0x45"
    cases = []
    for base in (0, 0xffffffff, 0xa58c3695):
        for p0, p2, p6 in product((0, 1), repeat=3):
            packed = p0 | (p2 << 2) | (p6 << 6)
            shift = 8 * byte
            expected = (base & ~(0x45 << shift)) | (packed << shift)
            cases.append(((base, p0, p2, p6), [expected]))
    check_registers([line], ["R0", "P0", "P2", "P6"], ["R1"], cases)


@pytest.mark.parametrize("backend", [CpuLifter, NVVMLifter])
@pytest.mark.parametrize("a_type,b_type", product(("S8", "U8"), repeat=2))
def test_idp_four_packed_bytes(backend, a_type, b_type):
    a, b, accum = bytes([128, 255, 0, 3]), bytes([254, 127, 17, 131]), 0xfffffff0
    dot = sum(int.from_bytes([x], signed=a_type == "S8") *
              int.from_bytes([y], signed=b_type == "S8") for x, y in zip(a, b))
    line = f"IDP.4A.{a_type}.{b_type} R3, R0, R1, R2"
    args = (int.from_bytes(a, "little"), int.from_bytes(b, "little"), accum)
    check_registers([line], ["R0", "R1", "R2"], ["R3"], [(args, [(accum + dot) & 0xffffffff])], backend)


@pytest.mark.parametrize("source,values", [
    ("S32", [(16777219, 16777218), (-16777219, -16777218), (-17, -17)]),
    ("U32", [(16777219, 16777218), (0xffffffff, 4294967040), (17, 17)]),
])
def test_i2f_toward_zero(source, values):
    cases = [([value], [float_bits(expected)]) for value, expected in values]
    check_registers([f"I2F.RZ.{source} R1, R0"], ["R0"], ["R1"], cases, CpuLifter)


def test_hfma2_preserves_subnormal_half_literals():
    line = "HFMA2 R0, -RZ, RZ, -5.9604644775390625e-08, 2.384185791015625e-07"
    check_registers([line], [], ["R0"], [([], [0x80010004])])


def test_mov_and_iadd_register_pairs_propagate_carry():
    lines = ["MOV.64 R4, R0", "IADD.64 R4, R4, R2"]
    cases = [((0xffffffff, 0x12345678, 1, 0x20), (0, 0x12345699)),
             ((0xffffffff, 0xffffffff, 1, 0), (0, 0))]
    check_registers(lines, ["R0", "R1", "R2", "R3"], ["R4", "R5"], cases)
