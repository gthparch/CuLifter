from types import SimpleNamespace

import pytest
from llvmlite import binding as llvm, ir

from lift.lifter_nvvm import NVVMLifter
from sir.basicblock import BasicBlock
from sir.function import Function
from sir.instruction import Instruction
from sir.operand import Operand
from transform.typeanalysis import TypeAnalysis


@pytest.mark.parametrize("low, high, count, phase", [
    (0x1FFFF4, 0xFFFFD000, 6, 1),   # NVJET's initially empty TMA stage
    (0x1FFFF4, 0x7FFFD000, 6, 0),
    (0x1FFF00, 0xFFFC0000, 128, 1),
    (0x1FFFFE, 0x7FFFF800, 1, 0),
])
def test_syncs_exch_init(low, high, count, phase):
    lifter = NVVMLifter(SimpleNamespace(arg_map={}, const_data={}))
    module = ir.Module()
    i32 = ir.IntType(32)
    lifter.SharedMem = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), 8),
                                         "shared_mem", addrspace=3)
    func = ir.Function(module, ir.FunctionType(ir.VoidType(), []), "init")
    builder = ir.IRBuilder(func.append_basic_block("entry"))
    operands = [Operand.parse(token) for token in ("URZ", "[UR6]", "UR10")]
    inst = Instruction(0, ["SYNCS", "EXCH", "64"], operands)
    kernel = Function("init")
    kernel.blocks = [BasicBlock("0", None, [inst])]
    TypeAnalysis().process_func(kernel)
    regs = {reg: ir.Constant(i32, value) for reg, value in
            (("UR6", 0), ("UR10", low), ("UR11", high))}
    lifter.lift_instruction(inst, builder, regs, {})
    builder.ret_void()

    llvm.initialize_all_targets()
    target = llvm.Target.from_triple("nvptx64-nvidia-cuda").create_target_machine(cpu="sm_90a")
    mod = llvm.parse_assembly(str(module))
    mod.verify()
    pm = llvm.create_new_module_pass_manager()
    pm.add_instruction_combine_pass()
    pm.run(mod, llvm.create_pass_builder(target, llvm.PipelineTuningOptions()))
    call, = (i for block in mod.get_function("init").blocks for i in block.instructions
             if i.opcode == "call")
    _, n, initial_phase, asm = call.operands
    assert (n.get_constant_value(), initial_phase.get_constant_value()) == (count, phase)
    assert "@leader mbarrier.init" in str(asm)
    assert "@advance mbarrier.arrive.shared::cta.b64 _, [$0], $1" in str(asm)
