import pytest

from parse.parser import SaSSParser


@pytest.mark.parametrize("format", ["nvdisasm", "cuobjdump"])
def test_annotated_brx_keeps_guard_and_register(format):
    branch = '@!UP0 BRX R6 (*"BRANCH_TARGETS .L_1,.L_2"*);'
    if format == "nvdisasm":
        text = f""".section .nv.info
.section .text.probe
.type probe,@function
probe:
/*0000*/ {branch}
.L_1:
/*0010*/ P2R.B1 R0, PR, R2, 0x4;
.L_2:
/*0020*/ EXIT;
"""
    else:
        text = f"""code for sm_90
Function : probe
/*0000*/ {branch} /* 0x0000000000000000 */
/*0010*/ P2R.B1 R0, PR, R2, 0x4; /* 0x0000000000000000 */
/*0020*/ EXIT; /* 0x0000000000000000 */
"""
    func, = SaSSParser(text, None).apply()
    brx, pack, exit_inst = func.instructions
    assert brx.opcodes == ["BRX"]
    assert (brx.pflag.reg, brx.pflag.prefix) == ("UP0", "!")
    target, = brx.operands
    assert [word.reg for word in target.reg_group] == ["R6", "R7"]
    assert pack.opcodes == ["P2R", "B1"] and pack.pflag is None
    assert [op.reg for op in pack.get_uses()[3:]] == ["P2"]
    assert exit_inst.opcodes == ["EXIT"]
