import ctypes as ct
import os
import re
import shutil

import pytest
from llvmlite import binding as llvm

from lift.lifter import UnsupportedInstructionException
from lift.lifter_nvvm import NVVMLifter
from transform.tmem_alloc import mark_full_tmem_alloc
from utils.elf import Elf
from .test_ldtm_functional import sm100_context
from .test_nvvm_sass_regressions import FIXTURES, lift_kernel, parse_kernel
from .test_utils import run_cmd


ALLOC_FREE = [
    "LDC.64 R2, c[0x0][0x380]",
    "UMOV UR5, 0x10",
    "VOTEU.ANY UP0, PT",
    "@UP0 UTCATOMSWS.FIND_AND_SET UPT, UR4, UR5",
    "IMAD.U32 R0, RZ, RZ, UR4",
    "STG.E [R2.64], R0",
    "SHF.L.U32 R0, R0, 0x5, RZ",
    "IMAD.MOV.U32 R3, RZ, RZ, 0x1ffff",
    "SHF.R.U32.HI R0, RZ, 0x5, R0",
    "SHF.L.U32 R0, R3, R0, RZ",
    "R2UR UR5, R0",
    "@UP0 ULOP3.LUT UR5, URZ, UR5, URZ, 0x33, !UPT",
    "@UP0 UTCATOMSWS.AND URZ, UR5",
]


@pytest.mark.parametrize("offset,axis,uniform,reciprocal", [
    ("0x2a0", "x", False, False), ("0x2a8", "z", True, False), ("0x2b4", "y", True, True)])
def test_cluster_table_reads_are_derived_from_launch_dimensions(offset, axis, uniform, reciprocal):
    idx = "UR5" if uniform else "R7"
    program = ["CS2R.32 R7, SR_CgaSize", "IMAD R7, R7, -0xa0, RZ"]
    if uniform: program.append("R2UR UR5, R7")
    program += [f"{'LDCU UR6' if uniform else 'LDC R6'}, c[0x0][{idx}+{offset}]",
                "LDC.64 R2, c[0x0][0x380]", "S2R R8, SR_VIRTUALSMID",
                "STG.E [R2.64], " + ("UR6" if uniform else "R6"), "STG.E [R2.64+0x4], R8"]
    ir = str(lift_kernel(program, 100))
    assert f"llvm.nvvm.read.ptx.sreg.cluster.nctaid.{axis}" in ir
    assert "llvm.nvvm.read.ptx.sreg.smid" in ir
    assert ("fdiv float" in ir) == reciprocal


@pytest.mark.parametrize("mode,scope", [("SELFCAST", ""), ("BROADCAST", ".multicast::cluster::all")])
def test_work_request_keeps_both_shared_addresses_and_issues_once(mode, scope):
    module = lift_kernel(["UMOV UR6, 0x180", "UMOV UR7, 0x40",
                          f"UGETNEXTWORKID.{mode} [UR6], [UR7]"], 100)
    ir = str(module)
    assert ir.count(f"clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes{scope}.b128") == 1
    assert "elect.sync" in ir
    assert "i32 384" in ir and "i32 64" in ir


def test_full_tmem_allocation_lowers_both_ends():
    module = lift_kernel(ALLOC_FREE, 100)
    text = str(module)
    for operation in ("alloc", "dealloc", "relinquish_alloc_permit"):
        assert text.count(f"tcgen05.{operation}.cta_group::1.sync.aligned") == 1
    assert text.count(", 512;") == 2


@pytest.mark.parametrize("old,new", [("0x10", "0x8"), ("0x1ffff", "0xffff"),
                                     ("0x33", "0x3c"), ("FIND_AND_SET", "FIND_AND_SET.ALIGN"),
                                     ("@UP0 UTCATOMSWS.AND", "@P0 UTCATOMSWS.AND")])
def test_other_sw_state_operations_remain_unsupported(old, new):
    lines = [line.replace(old, new) for line in ALLOC_FREE]
    func, = parse_kernel(lines, 100, "test_call", None).functions
    mark_full_tmem_alloc(func)
    sites = [inst for inst in func.instructions if inst.opcodes[0] == "UTCATOMSWS"]
    assert len(sites) == 2
    lifter = NVVMLifter(Elf(FIXTURES / "test_call.cubin"))
    for site in sites:
        with pytest.raises(UnsupportedInstructionException, match="SW-state"):
            lifter.lift_instruction(site, None, {}, {})


@pytest.mark.parametrize("pred", ["UP0", "UPT"])
def test_dense_mma_tracks_descriptor_high_words(pred):
    line = f"UTCIMMA gdesc[UR26], gdesc[UR24], tmem[UR20], tmem[UR16], idesc[UR17], {pred}"
    inst = parse_kernel([line], 100, "test_call", None).functions[0].instructions[0]
    a, b, dest, metadata, idesc, use_d = inst.get_uses()
    assert [[reg.reg for reg in op.reg_group] for op in (a, b)] == [["UR26", "UR27"], ["UR24", "UR25"]]
    assert [op.reg for op in (dest, metadata, idesc, use_d)] == ["UR20", "UR16", "UR17", pred]
    assert not inst.get_defs()


@pytest.fixture
def assemble_sm100a(tmp_path):
    if not all(shutil.which(tool) for tool in ("ptxas", "cuobjdump")):
        pytest.skip("ptxas and cuobjdump are required")
    if "sm_100a" not in run_cmd(["ptxas", "--help"], tmp_path):
        pytest.skip("ptxas lacks sm_100a")
    def assemble(lines):
        module = lift_kernel(lines, 100)
        llvm.initialize_all_targets()
        llvm.initialize_all_asmprinters()
        machine = llvm.Target.from_triple(module.triple).create_target_machine(cpu="sm_100a", features="+ptx87")
        ptx = tmp_path / "probe.ptx"
        ptx.write_text(machine.emit_assembly(module))
        cubin = ptx.with_suffix(".cubin")
        run_cmd(["ptxas", "-arch=sm_100a", ptx, "-o", cubin], tmp_path)
        return cubin
    return assemble


def test_work_request_round_trip(tmp_path, assemble_sm100a):
    cubin = assemble_sm100a(["UMOV UR6, 0x180", "UMOV UR7, 0x40",
                             "UGETNEXTWORKID.BROADCAST [UR6], [UR7]"])
    sass = run_cmd(["cuobjdump", "-sass", cubin], tmp_path)
    assert sass.count("UGETNEXTWORKID.BROADCAST") == 1


def test_sm100a_tmem_cubin_round_trip(tmp_path, assemble_sm100a):
    cubin = assemble_sm100a([
        *ALLOC_FREE[:6],
        "LDCU.64 UR8, c[0x0][0x380]", "LDCU.64 UR10, c[0x0][0x388]",
        "UMOV UR16, 0x0", "UMOV UR17, 0x84004a0", "UMOV UR18, 0x8",
        "LDCU UR19, c[0x0][0x390]",
        "UTCIMMA gdesc[UR8], gdesc[UR10], tmem[UR4], tmem[UR16], idesc[UR17], UP0",
        "UTCBAR [UR18], URZ", "UTCBAR.MULTICAST [UR18], URZ, UR19", *ALLOC_FREE[6:],
    ])
    sass = run_cmd(["cuobjdump", "--dump-sass", cubin], tmp_path)
    assert "UTCATOMSWS.FIND_AND_SET" in sass and "UTCATOMSWS.AND" in sass
    assert re.search(r"UTCIMMA gdesc\[UR\d+\], gdesc\[UR\d+\], tmem\[UR\d+\], tmem\[UR\d+\], idesc\[UR\d+\]", sass)
    assert re.search(r"UTCBAR \[UR\d+\], URZ", sass)
    assert re.search(r"UTCBAR.MULTICAST \[UR\d+\], URZ, UR\d+", sass)


def test_full_tmem_allocation_on_sm100(sm100_context, assemble_sm100a):
    cuda, check = sm100_context
    cubin = assemble_sm100a([*ALLOC_FREE[:5], "S2R R6, SR_LANEID",
                             "IMAD.WIDE.U32 R2, R6, 0x4, R2", *ALLOC_FREE[5:]])
    module, kernel = ct.c_void_p(), ct.c_void_p()
    check(cuda.cuModuleLoad(ct.byref(module), os.fsencode(cubin)))
    try:
        check(cuda.cuModuleGetFunction(ct.byref(kernel), module, b"test_call"))
        words = (ct.c_uint32 * 32)()
        size = ct.c_size_t(ct.sizeof(words))
        output, unused, count = ct.c_uint64(), ct.c_uint64(), ct.c_uint32()
        check(cuda.cuMemAlloc_v2(ct.byref(output), size))
        try:
            args = (ct.c_void_p * 3)(*(ct.addressof(v) for v in (output, unused, count)))
            for _ in range(2):
                check(cuda.cuMemsetD8_v2(output, 0xff, size))
                check(cuda.cuLaunchKernel(kernel, 1, 1, 1, 32, 1, 1, 0, None, args, None))
                check(cuda.cuMemcpyDtoH_v2(words, output, size))
                assert list(words) == [0] * 32
        finally:
            check(cuda.cuMemFree_v2(output))
    finally:
        check(cuda.cuModuleUnload(module))


@pytest.mark.parametrize("group,modifier", [(1, ""), (2, ".2CTA")])
def test_column_data_movement_and_fp16_survive_lifting(group, modifier):
    lines = [
        "LDCU.64 UR6, c[0x0][0x380]", "LDCU.64 UR8, c[0x0][0x388]",
        "UMOV UR4, 0", "UMOV UR10, 0", "UMOV UR11, 0x8100010",
        *[f"IMAD.MOV.U32 R{4+i}, RZ, RZ, {i+1}" for i in range(4)],
        f"UTCCP.T.S.128dp128bit{modifier} tmem[UR4], gdesc[UR6]",
        "STTM.x4 tmem[UR4], R4",
        f"UTCHMMA{modifier} gdesc[UR6], gdesc[UR8], tmem[UR4], tmem[UR10], idesc[UR11], UPT",
    ]
    text = str(lift_kernel(lines, 100))
    assert f"tcgen05.cp.cta_group::{group}.128x128b" in text
    assert "tcgen05.st.sync.aligned.32x32b.x4.b32" in text
    assert f"tcgen05.mma.cta_group::{group}.kind::f16" in text


@pytest.mark.parametrize("original,replacement,recognized", [
    ("", "", True),
    ("BRA.U !UP0, 0x400", "BRA.U UP0, 0x400", False),
    ("R5, RZ, RZ, 0xf", "R5, RZ, RZ, 0x7", False),
    ("LDS R0, [UR10+0x2000]", "LDS R0, [UR10+0x2004]", False),
    ("ULOP3.LUT UR4, URZ, UR4, URZ, 0x33", "ULOP3.LUT UR4, URZ, UR4, URZ, 0x3c", False),
])
def test_single_aligned_alloc_protocol(original, replacement, recognized):
    from pathlib import Path
    from parse.parser import SaSSParser
    from sir.module import Module
    from transform.tmem_column import recover_single_aligned

    source = (Path(__file__).parent / "sm100_optest/column_alloc_128.sass").read_text()
    if original:
        assert original in source
        source = source.replace(original, replacement)
    func, = Module("test", SaSSParser(source, None)).functions
    recover_single_aligned(func)
    assert sum(inst.opcodes == ["TMEMALLOC1"] for inst in func.instructions) == int(recognized)
    assert sum(inst.opcodes[-1] == "ALLOCATED" for inst in func.instructions) == 2 * int(recognized)
    assert sum(inst.opcodes[-1] == "ALIGNED_RELEASE" for inst in func.instructions) == int(recognized)


@pytest.mark.parametrize("middle,expected", [
    ([], True),
    (["UMOV UR15, 0x8100010"], True),
    (["@!UP0 UMOV UR15, 0x8100010"], True),
    (["UISETP.NE.U32.AND UP0, UPT, UR4, 1, UPT"], False),
    (["R2UR UR15, R9"], False),
])
def test_guarded_mma_descriptor_writers(middle, expected):
    from transform.tmem_column import recover_guarded_mma_constants

    lines = ["@UP0 UMOV UR14, 0x0", "@UP0 UMOV UR15, 0x8100490", *middle,
             "@UP0 UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR11], tmem[UR14], idesc[UR15], !UPT"]
    func, = parse_kernel(lines, 100, "test_call", None).functions
    recover_guarded_mma_constants(func)
    desc = func.instructions[-2].operands[4]
    assert desc.is_immediate == expected
    if expected:
        assert desc.immediate_value == (0x8100010 if middle == ["UMOV UR15, 0x8100010"] else 0x8100490)


@pytest.mark.parametrize("alternate,known", [("0x8100490", True), ("0x8100010", False)])
def test_guarded_descriptor_must_agree_at_branch_join(alternate, known):
    from transform.tmem_column import recover_guarded_mma_constants

    func, = parse_kernel([
        "@P0 BRA 0x30",
        "@UP0 UMOV UR15, 0x8100490",
        "BRA 0x40",
        f"@UP0 UMOV UR15, {alternate}",
        "@UP0 UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR11], tmem[UR14], idesc[UR15], !UPT",
    ], 100, "test_call", None).functions
    recover_guarded_mma_constants(func)
    assert func.instructions[4].operands[4].is_immediate == known


@pytest.mark.parametrize("transfer", [
    "CALL.REL.NOINC 0x30",
    "CALL.ABS.NOINC R2",
    "CALL.REL.NOINC R2 0x0",
    "BRXU UR4 0x0",
])
def test_guarded_descriptor_definition_bypassed_by_transfer(transfer):
    from transform.tmem_column import recover_guarded_mma_constants

    func, = parse_kernel([
        "@P0 BRA 0x50",
        "@UP0 UMOV UR15, 0x8100490",
        "NOP",
        "@UP0 UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR11], tmem[UR14], idesc[UR15], !UPT",
        "EXIT",
        transfer,
    ], 100, "test_call", None).functions
    recover_guarded_mma_constants(func)
    assert func.instructions[3].operands[4].is_uniform_reg


def test_guarded_descriptor_at_kernel_entry_with_backedge():
    from transform.tmem_column import recover_guarded_mma_constants

    func, = parse_kernel([
        "@UP0 UTCHMMA gdesc[UR4], gdesc[UR6], tmem[UR11], tmem[UR14], idesc[UR15], !UPT",
        "@UP0 UMOV UR15, 0x8100490",
        "BRA 0x0",
    ], 100, "test_call", None).functions
    recover_guarded_mma_constants(func)
    assert func.instructions[0].operands[4].is_uniform_reg
