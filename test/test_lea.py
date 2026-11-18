import ctypes
from types import SimpleNamespace

from llvmlite import binding as llvm, ir

from lift.lifter_nvvm import NVVMLifter
from sir.instruction import Instruction
from sir.operand import Operand


def test_subtract_shifted_64bit_offset():
    """Ampere GEMM uses a LEA / LEA.HI.X pair for base - (offset << 1)."""
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = ir.Module()
    i32, i64 = ir.IntType(32), ir.IntType(64)
    func = ir.Function(module, ir.FunctionType(i64, [i32] * 4), "address")
    builder = ir.IRBuilder(func.append_basic_block("entry"))
    regs = dict(zip(("R0", "R1", "R2", "R3"), func.args))
    for opcode, operands in (
        ("LEA", ("R4", "P0", "-R0", "R2", "0x1")),
        ("LEA.HI.X", ("R5", "~R0", "R3", "R1", "0x1", "P0")),
    ):
        ops = [Operand.parse(token) for token in operands]
        for op in ops:
            op.type_desc = "Bool" if op.is_predicate_reg else "UInt32"
        lifter.lift_instruction(Instruction(opcode, opcode.split("."), ops), builder, regs, {})
    high = builder.shl(builder.zext(regs["R5"], i64), ir.Constant(i64, 32))
    builder.ret(builder.or_(high, builder.zext(regs["R4"], i64)))

    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    target = llvm.Target.from_default_triple().create_target_machine()
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    with llvm.create_mcjit_compiler(mod, target) as engine:
        engine.finalize_object()
        signature = ctypes.CFUNCTYPE(ctypes.c_uint64, *([ctypes.c_uint32] * 4))
        address = signature(engine.get_function_address("address"))
        for offset in (0, 1, 0x80000000, 0x12345678FFFFFFFF):
            for base in (0, 0x100000000, 0x12345678FFFFFFFF):
                result = address(offset & 0xFFFFFFFF, offset >> 32, base & 0xFFFFFFFF, base >> 32)
                assert result == (base - (offset << 1)) % (1 << 64)
