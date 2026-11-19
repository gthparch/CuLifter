import pytest
from llvmlite import binding as llvm

from .test_constant_bank import lift_const_bank_kernel


@pytest.mark.parametrize("opcode, left, expected", [
    ("HSET2.GEU.AND", 0x00004000, 0x0000FFFF),
    ("HSET2.LEU.AND", 0x00004000, 0xFFFF0000),
    ("HSET2.GEU.AND", 0x40007E00, 0xFFFFFFFF),
    ("HSET2.GE.AND", 0x40007E00, 0xFFFF0000),
    ("HSET2.BF.GEU.AND", 0x40007E00, 0x3C003C00),
    ("HSET2.EQU.XOR", 0x40007E00, 0xFFFF0000),
    ("HSET2.NAN.AND", 0x40007E00, 0x0000FFFF),
    ("HSET2.NUM.AND", 0x40007E00, 0xFFFF0000),
])
def test_hset2_comparison_and_output_format(opcode, left, expected):
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        f"MOV R4, {left:#x}", "MOV R5, 0x3c003c00",
        f"{opcode} R6, R4, R5, PT",
        "STG.E [R2.64], R6",
    ], 80, [(0x160, 8)])

    llvm.initialize_all_targets()
    machine = llvm.Target.from_triple(compiled.triple).create_target_machine(cpu="sm_80")
    passes = llvm.create_new_module_pass_manager()
    passes.add_instruction_combine_pass()
    passes.run(compiled, llvm.create_pass_builder(machine, llvm.PipelineTuningOptions()))
    [store] = [inst for block in compiled.get_function("test_call").blocks
               for inst in block.instructions if inst.opcode == "store" and "addrspace(1)" in str(inst)]
    value, _ = store.operands
    assert value.get_constant_value() == expected
