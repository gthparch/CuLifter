from io import StringIO

from llvmlite import binding as llvm

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.basicblock import BasicBlock
from sir.instruction import Instruction
from sir.module import Module
from sir.operand import Operand
from transform.cfg_simplify import CFGSimplify
from utils.asm import Asm


def test_lift_predicated_call_through_trampoline():
    sass = """code for sm_80
Function : probe
/*0000*/ S2R R0, SR_TID.X;                    /* 0x0 */
/*0010*/ ISETP.NE.AND P0, PT, R0, RZ, PT;    /* 0x0 */
/*0020*/ @P0 CALL.REL.NOINC 0x40;             /* 0x0 */
/*0030*/ EXIT;                               /* 0x0 */
/*0040*/ BRA 0x50;                           /* 0x0 */
/*0050*/ EXIT;                               /* 0x0 */
"""
    elf = Asm(sass)
    module = Module("call", SaSSParser(sass, elf=elf))
    lifter = NVVMLifter(elf)
    output = StringIO()
    lifter.lift_module(module, output)

    assert not lifter.placeholder_log
    lowered = llvm.parse_assembly(output.getvalue())
    lowered.verify()
    blocks = {block.name: block for block in lowered.get_function("probe").blocks}
    assert "BB_0040" not in blocks
    branch, = blocks["BB_80000020"].instructions
    target, = branch.operands
    assert branch.opcode == "br"
    assert target.name == "BB_0050"


def test_redirecting_call_fallthrough_keeps_register_relative_offset():
    call = Instruction(
        "0010", ["CALL", "REL", "NOINC"], [Operand.parse("R6"), Operand.parse("0x20")]
    )
    caller = BasicBlock("0010", None, [call])
    fallthrough = BasicBlock("0020", None, [])
    successor = BasicBlock("0030", None, [])
    caller.add_succ(fallthrough)
    fallthrough.add_pred(caller)

    CFGSimplify()._redirect_edge(caller, fallthrough, successor)

    assert caller.succs == [successor]
    register, offset = call.operands
    assert register.reg == "R6"
    assert offset.immediate_value == 0x20
