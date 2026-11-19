"""Assemble original PTX, recover from the actual cubin, and reassemble for the same SM."""
import ctypes
import io
import lzma
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from llvmlite import binding as llvm

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.module import Module
from transform.tmem_pair import recover_tmem_pair
from utils.asm import Asm
from utils.elf import Elf

TARGETS = ("sm_100", "sm_100a", "sm_120", "sm_120a")
CANCEL = "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes"
PAIR = "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"
QUERY = """
    ld.global.b128 response, [out];
    clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 ok, response;
    selp.u32 v, 1, 0, ok;
    st.global.u32 [out], v;
    @!ok ret;
    clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x,y,z,_}, response;
    st.global.v2.b32 [out+16], {x,y};
    st.global.b32 [out+24], z;
"""


@pytest.fixture(scope="module")
def toolkit():
    if not all(shutil.which(t) for t in ("ptxas", "cuobjdump")):
        pytest.skip("requires ptxas and cuobjdump")
    if "sm_120a" not in subprocess.check_output(["ptxas", "--help"], text=True):
        pytest.skip("requires a Blackwell-capable ptxas")
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()


def assemble(tmp_path, arch, body):
    ptx = tmp_path / "input.ptx"
    ptx.write_text(f""".version 8.7
.target {arch}
.address_size 64
.visible .entry probe(.param .u64 output, .param .u32 slot, .param .u32 bar) {{
    .reg .u64 out; .reg .b32 r,b,x,y,z,v; .reg .b128 response; .reg .pred ok;
    ld.param.u64 out,[output]; ld.param.u32 r,[slot]; ld.param.u32 b,[bar];
    {body}
    ret;
}}
""")
    cubin = ptx.with_suffix(".cubin")
    subprocess.run(["ptxas", "-arch=" + arch, ptx, "-o", cubin], check=True)
    return cubin


def disassemble(cubin):
    return subprocess.check_output(["cuobjdump", "-sass", cubin], text=True)


def lift(sass, cubin):
    elf = Elf(cubin) if cubin else Asm(sass)
    module = Module("cluster_probe", SaSSParser(sass, elf))
    out = io.StringIO()
    lifter = NVVMLifter(elf)
    lifter.lift_module(module, out)
    assert not lifter.placeholder_log
    compiled = llvm.parse_assembly(out.getvalue())
    compiled.verify()
    return compiled


def reassemble(compiled, arch, tmp_path):
    machine = llvm.Target.from_triple(compiled.triple).create_target_machine(cpu=arch, features="+ptx87")
    ptx = tmp_path / "recovered.ptx"
    ptx.write_text(machine.emit_assembly(compiled))
    cubin = ptx.with_suffix(".cubin")
    subprocess.run(["ptxas", "-arch=" + arch, ptx, "-o", cubin], check=True)
    return disassemble(cubin)


@pytest.mark.parametrize("arch, multicast", [(a, False) for a in TARGETS]
                         + [(a, True) for a in ("sm_100a", "sm_120a")])
def test_cancellation_delivery(toolkit, tmp_path, arch, multicast):
    op = CANCEL + (".multicast::cluster::all" if multicast else "") + ".b128"
    binary = assemble(tmp_path, arch, op + " [r], [b];")
    source = disassemble(binary)
    form = "BROADCAST" if multicast else "SELFCAST"
    assert source.count("UGETNEXTWORKID." + form) == 1
    compiled = lift(source, binary)
    text = str(compiled)
    assert text.count(op + " [$0], [$1]") == 1
    assert "elect.sync" in text
    assert re.search(r'call i1 @llvm.nvvm.vote.any.sync\(i32 %branch_mask, i1 %[^)]+\)', text)
    assert re.search(r'br i1 %branch_any,', text)
    assert reassemble(compiled, arch, tmp_path).count("UGETNEXTWORKID." + form) == 1


@pytest.mark.parametrize("arch", TARGETS)
@pytest.mark.parametrize("coordinates", ["vector", "scalar"])
def test_query_coordinates_and_failed_request(toolkit, tmp_path, arch, coordinates):
    query = QUERY
    if coordinates == "scalar":
        vector = "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {x,y,z,_}, response;"
        axes = [f"clusterlaunchcontrol.query_cancel.get_first_ctaid::{axis}.b32.b128 {axis}, response;"
                for axis in "xyz"]
        query = query.replace(vector, "\n".join(axes))
    binary = assemble(tmp_path, arch, query)
    compiled = lift(disassemble(binary), binary)
    reassemble(compiled, arch, tmp_path)
    # The query-only kernel contains integer ops and global loads/stores.
    # Execute its actual recovered body on the CPU, including the early exit.
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    machine = llvm.Target.from_default_triple().create_target_machine()
    compiled.triple = llvm.get_default_triple()
    compiled.data_layout = str(machine.target_data)
    u32 = ctypes.c_uint32
    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        call = ctypes.CFUNCTYPE(None, ctypes.c_uint64, u32, u32)(engine.get_function_address("probe"))
        for tag in (0, 1, 0xffffffff, 0xfffffffe):
            for x, y, z in [(0, 0, 0), (0x89abcdef, 0x1234, 0x9abc), (42, 0xffff, 0xffff)]:
                words = (u32 * 8)(x, y | (z << 16), tag, 0xa5a5a5a5, 91, 92, 93, 94)
                expected = list(words)
                expected[0] = tag & 1
                if tag & 1:
                    expected[4:7] = [x, y, z]
                call(ctypes.addressof(words), 0, 0)
                assert list(words) == expected


@pytest.mark.parametrize("columns", ["32", "64", "128", "256", "512", "b"])
def test_tmem_collective_at_warp_entry(toolkit, tmp_path, columns):
    binary = assemble(tmp_path, "sm_100a", f"{PAIR} [r], {columns};")
    sass = disassemble(binary)
    compiled = lift(sass, binary)
    text = str(compiled)
    assert text.count(PAIR) == 1
    assert "elect.sync" not in text
    assert "pair_dst" in text
    if columns == "b":
        assert "pair_columns" in text
    else:
        assert re.search(rf'{PAIR}[^\n]*i32 {columns}\)', text)
    assert reassemble(compiled, "sm_100a", tmp_path).count("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN") == 2


@pytest.mark.parametrize("columns", [32, 512])
def test_allocation_preserves_subsequent_shared_read_and_output(toolkit, tmp_path, columns):
    binary = assemble(tmp_path, "sm_100a", f"{PAIR} [r], {columns}; ld.shared.u32 x,[r]; st.global.u32 [out],x;")
    compiled = lift(disassemble(binary), binary)
    sass = reassemble(compiled, "sm_100a", tmp_path)
    assert str(compiled).count(PAIR) == 1
    assert "STG.E" in sass


@pytest.mark.parametrize("columns", ["32", "64", "512", "b"])
def test_static_shared_destination(toolkit, tmp_path, columns):
    binary = assemble(tmp_path, "sm_100a", f".shared .align 16 .b32 local[4]; "
                      f"mov.u32 r,local; {PAIR} [r],{columns};")
    compiled = lift(disassemble(binary), binary)
    assert str(compiled).count(PAIR) == 1
    assert "UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN" in reassemble(compiled, "sm_100a", tmp_path)


def paired_lifecycle(columns):
    return f"""
        {PAIR} [r],{columns};
        ld.shared.b32 x,[r];
        mov.u32 y,%ctaid.x;
        mov.u32 z,%tid.x;
        mad.lo.u32 y,y,32,z;
        mad.wide.u32 out,y,4,out;
        st.global.b32 [out],x;
        tcgen05.dealloc.cta_group::2.sync.aligned.b32 x,{columns};
        tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;
    """


@pytest.mark.parametrize("columns", [32, 64, 512])
def test_paired_allocation_and_release_with_per_lane_output(toolkit, tmp_path, columns):
    binary = assemble(tmp_path, "sm_100a", paired_lifecycle(columns))
    compiled = lift(disassemble(binary), binary)
    text = str(compiled)
    assert text.count(PAIR) == 1
    assert text.count("tcgen05.dealloc.cta_group::2.sync.aligned") == 1
    assert text.count("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned") == 1
    assert text.index(PAIR) < text.index("store i32", text.index(PAIR)) < text.index("tcgen05.dealloc")
    sass = reassemble(compiled, "sm_100a", tmp_path)
    for op in ("UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN", "STG.E", "UTCATOMSWS.AND", "UVIRTCOUNT.DEALLOC.SMPOOL"):
        assert op in sass


@pytest.mark.parametrize("where", ["consumer_mask", "wrong_slot", "wrong_trap"])
def test_unmatched_paired_release_is_never_consumed(toolkit, tmp_path, where):
    binary = assemble(tmp_path, "sm_100a", paired_lifecycle(512))
    source = disassemble(binary)
    # Mutate only after the second WARPSYNC (the end of the allocation).
    offset = [m.end() for m in re.finditer(r"WARPSYNC.ALL", source)][1]
    head, tail = source[:offset], source[offset:]
    if where == "consumer_mask":
        tail, count = re.subn(r"(IMAD.MOV.U32 R\d+, RZ, RZ, )0xffff", r"\g<1>0xff", tail, count=1)
    elif where == "wrong_slot":
        tail, count = re.subn(r"(LDCU UR\d+, c\[0x0\]\[)0x388", r"\g<1>0x38c", tail, count=1)
    else:
        tail, count = re.subn(r"BPT.TRAP 0x1", "NOP", tail, count=1)
    assert count == 1
    func, = Module("probe", SaSSParser(head + tail, Elf(binary))).functions
    recover_tmem_pair(func)
    assert all(i.opcodes[0] not in ("TMEMALLOC2", "TMEMFREE2") for i in func.instructions)
    assert any(i.opcodes == ["UTCATOMSWS", "AND"] for i in func.instructions)


@pytest.mark.parametrize("arch,operation", [
    ("sm_100", CANCEL + ".multicast::cluster::all.b128 [r],[b];"),
    ("sm_120", CANCEL + ".multicast::cluster::all.b128 [r],[b];"),
    *((a, PAIR + " [r],32;") for a in ("sm_100", "sm_120", "sm_120a")),
])
def test_target_restrictions(toolkit, tmp_path, arch, operation, capfd):
    with pytest.raises(subprocess.CalledProcessError):
        assemble(tmp_path, arch, operation)
    assert "target" in capfd.readouterr().err.lower()


def test_pattern_tracks_relocated_edges_registers_and_parameter_slot(toolkit, tmp_path):
    binary = assemble(tmp_path, "sm_100a", f"{PAIR} [r], 32;")
    sass = disassemble(binary)
    sass = re.sub(r"\bR(\d+)\b", lambda m: "R" + str(32 + int(m[1])), sass)
    sass = re.sub(r"/\*([0-9a-f]{4})\*/", lambda m: f"/*{int(m[1], 16) + 0x100:04x}*/", sass)
    sass = re.sub(r"(\bBRA(?:\.U)? (?:!?UP\d+, )?)0x([0-9a-f]+)",
                  lambda m: m[1] + hex(int(m[2], 16) + 0x100), sass)
    sass = sass.replace("c[0x0][0x388]", "c[0x0][0x38c]")
    func, = Module("probe", SaSSParser(sass, Elf(binary))).functions
    recover_tmem_pair(func)
    alloc, = (inst for inst in func.instructions if inst.opcodes == ["TMEMALLOC2"])
    assert int(alloc.id, 16) == 0x110
    dst, count = alloc.get_uses()
    assert dst.offset == 0x38c and count.immediate_value == 32


@pytest.mark.parametrize("old,new", [
    ("NANOSLEEP 0x64", "NANOSLEEP 0x10"),
    ("STAS [R6.64], R4", "STAS [R6.64], R5"),
    ("BRA.U !UP0, 0x110", "BRA.U !UP0, 0x170"),
    ("EXIT ;", "STG.E [R2.64], R4 ;"),
    ("BRA 0x4d0", "BRA 0xe0"),
])
def test_partial_or_externally_observed_protocol_is_not_folded(toolkit, tmp_path, old, new):
    binary = assemble(tmp_path, "sm_100a", f"{PAIR} [r], 32;")
    original = disassemble(binary)
    assert old in original
    sass = original.replace(old, new, 1)
    func, = Module("probe", SaSSParser(sass, Elf(binary))).functions
    recover_tmem_pair(func)
    assert sum(i.opcodes[0] == "UTCATOMSWS" for i in func.instructions) == 2
    assert not any(i.opcodes == ["TMEMALLOC2"] for i in func.instructions)


@pytest.mark.parametrize("arch", TARGETS)
def test_nvdisasm_target_suffix(toolkit, tmp_path, arch):
    if not shutil.which("nvdisasm"):
        pytest.skip("requires nvdisasm")
    binary = assemble(tmp_path, arch, f"{PAIR} [r], 32;" if arch == "sm_100a" else QUERY)
    listing = subprocess.check_output(["nvdisasm", "-hex", binary], text=True)
    assert SaSSParser(listing, Elf(binary)).isa == arch
    compiled = lift(listing, None)
    if arch == "sm_100a":
        assert str(compiled).count(PAIR) == 1


@pytest.mark.skipif(os.environ.get("CULIFTER_TEST_SIMX") != "1",
                    reason="set CULIFTER_TEST_SIMX=1 in a configured CuPBoP/Vortex environment")
@pytest.mark.parametrize("arch,case", [(a, "query") for a in TARGETS] + [("sm_100a", "pair")])
def test_simx_cpu_comparison(toolkit, tmp_path, arch, case):
    examples = Path(os.environ["CuPBoP_PATH"]) / "examples"
    regression = Path(os.environ["VORTEX_PATH"]) / "tests/regression"
    if case == "pair" and "-DVX_CFG_TCU_TMEM_ENABLE" not in os.environ["VORTEX_CONFIGS"].split():
        pytest.skip("the paired allocation needs a TMEM-enabled SimX build")
    binary = assemble(tmp_path, arch, QUERY if case == "query" else paired_lifecycle(512))
    compiled = lift(disassemble(binary), binary)

    # lift.mk links the host runtime relative to the example directory.
    (tmp_path / "build").symlink_to(examples.parent / "build", target_is_directory=True)
    work = tmp_path / "examples" / "probe"
    work.mkdir(parents=True)
    source = Path(__file__).parent / "sm100_optest/cluster_recovery_simx.cu"
    shutil.copyfile(source, work / "kernel.cu")
    (work / "lift_host.h").symlink_to(examples / "lift_host.h")
    (work / "raw.ll.xz").write_bytes(lzma.compress(str(compiled).encode()))
    (work / "Makefile").write_text(f"KERNEL_CU = kernel.cu\ninclude {examples}/lift.mk\n")
    env = dict(os.environ, VORTEX_DRIVER="simx", VORTEX_SCHEDULE_FLAG="2", VORTEX_LOCALMEM_FLAG="1",
               CUPBOP_LIFT_OPT="1", CLANG_NO_DEFAULT_CONFIG="1")
    with tempfile.TemporaryDirectory(prefix="culifter_cluster_", dir=regression) as cell:
        cmd = ["make", "-j1", f"VX3_CELL={cell}"]
        with (work / "build.log").open("w") as log:
            status = subprocess.run([*cmd, "build"], cwd=work, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=240)
        assert status.returncode == 0, (work / "build.log").read_text()[-4000:]
        if case == "pair":
            # Supply the 2x1x1 geometry to CuPBoP's by-name kernel launcher.
            (work / "cupbop_cluster_dims.txt").write_text("probe 2 1 1 0\n")
        run = subprocess.run([*cmd, "run", "RUN_ARGS=" + case], cwd=work, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
        assert run.returncode == 0 and "PASSED CPU comparison" in run.stdout, run.stdout
