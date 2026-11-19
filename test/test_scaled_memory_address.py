import pytest

from parse.parser import SaSSParser
from sir.module import Module
from transform.build_cfg import BuildCFG
from transform.opmod_transform import OpModTransform


def lower_address(instruction):
    sass = f"""code for sm_80
Function : probe
/*0000*/ {instruction} ; /* 0x0000000000000000 */
/*0010*/ EXIT ; /* 0x0000000000000000 */
"""
    module = Module("probe", SaSSParser(sass, None))
    BuildCFG().apply(module)
    OpModTransform().apply(module)
    func, = module.functions
    return func.blocks[0].instructions[:-1]


@pytest.mark.parametrize("scale, shift", [(2, 1), (4, 2), (8, 3), (16, 4)])
def test_scaled_shared_address_preserves_displacement(scale, shift):
    shl, load = lower_address(f"LDS R5, [R4.X{scale}+0x20]")
    assert shl.opcodes == ["SHL"]
    base, amount = shl.get_uses()
    assert base.reg == "R4"
    assert amount.immediate_value == shift

    address, = load.get_uses()
    assert address.reg == shl.get_defs()[0].reg
    assert address.suffix is None
    assert address.offset_value == 0x20


def test_scaled_shared_address_adds_index_after_scaling_base():
    shl, add, store = lower_address("STS [R4.X8+UR5-0x20], R6")
    assert shl.opcodes == ["SHL"]
    base, amount = shl.get_uses()
    assert (base.reg, amount.immediate_value) == ("R4", 3)

    assert add.opcodes == ["IADD3"]
    assert [arg.reg for arg in add.get_uses()] == [shl.get_defs()[0].reg, "UR5", "RZ"]

    address, value = store.get_uses()
    assert address.reg == add.get_defs()[0].reg
    assert address.index_reg is None
    assert address.offset_value == -0x20
    assert value.reg == "R6"
