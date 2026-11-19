import pytest
from llvmlite import binding as llvm

from .test_constant_bank import lift_const_bank_kernel


@pytest.mark.parametrize("opcode, left, right, addend, expected", [
    ("HFMA2", 0x5C015C01, 0x3BFE3BFE, 0xDC00DC00, 0x8C008C00),
    ("HFMA2.BF16_V2", 0x41814181, 0x3F7E3F7E, 0xC180C180, 0xBA80BA80),
])
def test_hfma2_rounds_only_after_addition(opcode, left, right, addend, expected):
    # Rounding the products first would cancel to zero in both formats.
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        f"MOV R4, {left:#x}", f"MOV R5, {right:#x}", f"MOV R6, {addend:#x}",
        f"{opcode} R7, R4, R5, R6", "STG.E [R2.64], R7",
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
