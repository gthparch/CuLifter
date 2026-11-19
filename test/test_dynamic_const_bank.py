import ctypes

import pytest

from .test_constant_bank import lift_const_bank_kernel, run_with_param_bits


@pytest.mark.parametrize("displacement", [0, 4, 12])
def test_banked_load_with_index_is_not_a_special_register(displacement):
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        "MOV R4, c[0x0][0x168]",
        f"LDC R6, c[0x0][R4+{displacement:#x}]",
        "STG.E [R2.64], R6",
    ], 80, [(0x160, 8), (0x168, 8)])
    assert "llvm.nvvm.read.ptx.sreg" not in str(compiled.get_function("test_call"))
    result = ctypes.c_uint32()
    expected = 0xABCDEF90
    run_with_param_bits(compiled, result, (expected << 32) | (0x16c - displacement))
    assert result.value == expected


@pytest.mark.parametrize("address, register", [("0x0", "ntid.x"), ("RZ+0xc", "nctaid.x")])
def test_fixed_constant_bank_special_register_mapping(address, register):
    compiled = lift_const_bank_kernel([
        "MOV R2, c[0x0][0x160]", "MOV R3, c[0x0][0x164]",
        f"LDC R6, c[0x0][{address}]", "STG.E [R2.64], R6",
    ], 80, [(0x160, 8)])
    assert f"@llvm.nvvm.read.ptx.sreg.{register}(" in str(compiled.get_function("test_call"))
