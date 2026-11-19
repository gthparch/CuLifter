import ctypes
import io
import re
from pathlib import Path

import pytest
from llvmlite import binding as llvm

from lift.lifter_nvvm import NVVMLifter
from parse.parser import SaSSParser
from sir.module import Module
from transform.build_cfg import BuildCFG
from transform.desc_const_map import RecordDescriptorConsts
from utils.elf import Elf


FIXTURES = Path(__file__).parent / "sm100_optest"


def parse_kernel(lines, sm, name, elf):
    sass = [f"code for sm_{sm}", f"Function : {name}"]
    for pc, line in enumerate([*lines, "EXIT"]):
        sass.append(f"/*{16 * pc:04x}*/ {line} ; /* 0x0000000000000000 */")
    return Module("test", SaSSParser("\n".join(sass), elf))


def lift_kernel(lines, sm, name="test_call", elf_file=None):
    """Use the checked-in three-argument kernel ABI and the full NVVM pass pipeline."""
    elf = Elf(elf_file or FIXTURES / f"test_call_sm{sm}.elf")
    if name != "test_call":
        elf.arg_map[name] = elf.arg_map["test_call"]
    module = parse_kernel(lines, sm, name, elf)
    out = io.StringIO()
    lifter = NVVMLifter(elf)
    lifter.lift_module(module, out)
    assert not lifter.placeholder_log
    compiled = llvm.parse_assembly(out.getvalue())
    compiled.verify()
    return compiled


def lift_ptx(lines, sm, **kwargs):
    compiled = lift_kernel(lines, sm, **kwargs)
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    target = llvm.Target.from_triple(compiled.triple)
    return target.create_target_machine(cpu=f"sm_{sm}", features="+ptx87").emit_assembly(compiled)


@pytest.mark.parametrize("modifier,cache,size", [
    ("", "ca", 4), (".32", "ca", 4), (".64", "ca", 8), (".128", "cg", 16),
])
def test_ldgsts_cache_hint_and_access_width(modifier, cache, size):
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x210]",
        "LDC R4, c[0x0][0x220]",
        f"LDGSTS.E.BYPASS.LTC128B{modifier} [R4+{size:#x}], [R2.64]",
    ], 90)
    assert re.search(rf"cp\.async\.{cache}\.shared\.global \[[^]]+\], \[[^]]+\], {size};", ptx)


def test_ldgsts_and_syncs_fold_the_uniform_index_into_the_address():
    def shared_offsets(data, barrier, index):
        ptx = lift_ptx([
            f"MOV R6, {data:#x}", f"MOV R7, {barrier:#x}", "UMOV UR4, 0x24",
            "LDC.64 R2, c[0x0][0x210]", f"LDGSTS.E [R6{index}], [R2.64]",
            f"SYNCS.ARRIVE.TRANS64.RED.A1T0 RZ, [R7{index}], RZ",
        ], 90)
        pointers = [re.search(r"cp\.async\.ca\.shared\.global \[(%rd\d+)\]", ptx)[1],
                    re.search(r"mbarrier\.arrive\.shared::cta\.b64 s, \[(%rd\d+)\]", ptx)[1]]
        return [int(re.search(rf"add\.s64\s+{reg}, %rd\d+, (-?\d+);", ptx)[1]) for reg in pointers]

    direct = shared_offsets(0x224, 0x364, "")
    assert direct[1] - direct[0] == 0x140
    assert shared_offsets(0x200, 0x340, "+UR4") == direct


def test_ldcu_descriptor_tracks_each_reaching_constant_bank_load():
    module = parse_kernel([
        "LDCU.64 UR4, c[0x2][0x40]", "LDG.E R6, desc[UR4][R2.64]",
        "LDCU.64 UR4, c[0x0][0x358]", "LDG.E R7, desc[UR4][R2.64]",
    ], 100, "test_call", None)
    BuildCFG().apply(module)
    RecordDescriptorConsts().apply(module)
    func, = module.functions
    addresses = [inst.get_uses()[0] for block in func.blocks for inst in block.instructions
                 if inst.opcodes[0] == "LDG"]
    assert [address.desc_const for address in addresses] == [(2, 0x40), (0, 0x358)]
    assert "UR4" not in func.desc_const_map


def test_cluster_barriers_survive_around_stas():
    ptx = lift_ptx([
        "MOV R6, 0x02000100", "MOV R7, 0x05000200", "MOV R8, 0x438",
        "UCGABAR_ARV", "STAS [R6.64+0x10], R8", "UCGABAR_WAIT",
    ], 90)
    arrive = ptx.index("barrier.cluster.arrive.aligned;")
    wait = ptx.index("barrier.cluster.wait.aligned;")
    body = ptx[arrive:wait]
    dest, barrier = re.findall(r"mapa\.shared::cluster\.u32 [db], (%\w+), (%\w+);", body)
    for (address, rank), offset, cta in [(dest, 0x110, 2), (barrier, 0x200, 5)]:
        assert re.search(rf"add\.s32\s+{address}, %\w+, {offset};", body)
        assert re.search(rf"mov\.b32\s+{rank}, {cta};", body)
    value = re.search(r"st\.async\.weak\.shared::cluster\.mbarrier::complete_tx::bytes\.b32 \[d\], (%\w+), \[b\];", body)[1]
    assert re.search(rf"mov\.b32\s+{value}, 1080;", body)


@pytest.mark.parametrize("sm,elf_file", [
    (100, "test_call.cubin"), (100, "test_call_sm100.elf"), (120, "test_call_sm120.elf"),
])
def test_blackwell_driver_slots(sm, elf_file):
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x380]", "LDCU.64 UR4, c[0x0][0x348]",
        "LDC R6, c[0x0][0x36c]",
        "STG.E [R2.64], UR4", "STG.E [R2.64+0x4], UR5", "STG.E [R2.64+0x8], R6",
    ], sm, elf_file=FIXTURES / elf_file)
    assert "cvta.param.u64" in ptx and "test_call_param_0" in ptx
    assert "%is_explicit_cluster" in ptx


@pytest.mark.parametrize("name", ["nvjet_sm90_probe", "test_call"])
def test_hopper_nvjet_parameter_base(name):
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x210]", "LDC.64 R4, c[0x0][0xc0]",
        "STG.E.64 [R2.64], R4",
    ], 90, name=name)
    if name.startswith("nvjet"):
        assert f"{name}_param_0" in ptx and "cvta.param.u64" in ptx
        assert re.search(r"add\.s64\s+[^;]+, -528;", ptx)
    else:
        assert "cvta.param.u64" not in ptx


@pytest.mark.parametrize("opcode,ptx_op", [
    ("QMMA.16832.F32.E4M3.E5M2", "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32"),
    ("HMMA.16816.F32.BF16", "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"),
])
def test_mma_register_fragments(opcode, ptx_op):
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x380]",
        "LDG.E.128 R8, [R2.64]", "LDG.E.64 R12, [R2.64+0x10]",
        "LDG.E.128 R16, [R2.64+0x20]",
        f"{opcode} R16, R8, R12, R16", "STG.E.128 [R2.64+0x30], R16",
    ], 120)
    match = re.search(re.escape(ptx_op) + r"\s+([^;]+);", ptx)
    assert match
    dest, a, b, accum = [re.findall(r"%\w+", group)
                         for group in re.findall(r"\{([^}]+)\}", match[1])]
    assert len(set(dest)) == 4
    for regs, width, offset in [(a, 4, ""), (b, 2, "+16"), (accum, 4, "+32")]:
        load = re.search(rf"ld\.global\.v{width}\.b32\s+\{{([^}}]+)\}}, \[%rd\d+{re.escape(offset)}\];", ptx)
        assert load and re.findall(r"%\w+", load[1]) == regs
    assert "st.global" in ptx


def test_qmma_predicate_controls_all_four_accumulators():
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x380]", "LDCU UR6, c[0x0][0x390]",
        "UISETP.NE.AND UP0, UPT, UR6, URZ, UPT",
        "LDG.E.128 R8, [R2.64]", "LDG.E.64 R12, [R2.64+0x10]",
        "LDG.E.128 R16, [R2.64+0x20]",
        "QMMA.16832.F32.E5M2.E4M3 R16, R8, R12, R16, UP0",
        "STG.E.128 [R2.64+0x30], R16",
    ], 120)
    accum = re.search(r"mma\.sync[^;]+\{([^}]+)\};", ptx)[1].split(",")
    inputs = re.search(r"ld\.global\.v4\.b32\s+\{([^}]+)\}, \[%\w+\+32\];", ptx)[1].split(",")
    pred = re.search(r"setp\.ne\.b32\s+(%p\d+),", ptx)[1]
    assert len(accum) == len(inputs) == 4
    for out, value in zip(accum, inputs):
        assert re.search(rf"selp\.f32\s+{out.strip()}, {value.strip()}, 0[fF]00000000, {pred};", ptx)


@pytest.mark.parametrize("sign", ["S32", "U32"])
def test_nvvm_integer_to_float_toward_zero(sign):
    ptx = lift_ptx([
        "LDC.64 R2, c[0x0][0x380]", "LDC R4, c[0x0][0x390]",
        f"I2F.RZ.{sign} R6, R4", "STG.E [R2.64], R6",
    ], 100)
    assert f"cvt.rz.f32.{sign.lower()}" in ptx


@pytest.mark.parametrize("opcode,reg", [("LDC", "R"), ("ULDC", "UR"), ("LDCU", "UR")])
@pytest.mark.parametrize("width", [64, 128])
def test_wide_constant_load_returns_each_word_in_order(opcode, reg, width):
    words = width // 32
    lines = ["LDC.64 R2, c[0x0][0x380]", f"{opcode}.{width} {reg}8, c[0x0][0x380]"]
    lines.extend(f"STG.E [R2.64+0x{4 * i:x}], {reg}{8 + i}" for i in range(words))
    compiled = lift_kernel(lines, 100)
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    compiled.triple = llvm.get_default_triple()
    machine = llvm.Target.from_default_triple().create_target_machine()
    compiled.data_layout = str(machine.target_data)
    u64, u32 = ctypes.c_uint64, ctypes.c_uint32
    with llvm.create_mcjit_compiler(compiled, machine) as engine:
        engine.finalize_object()
        call = ctypes.CFUNCTYPE(None, u64, u64, u32)(engine.get_function_address("test_call"))
        out = (u32 * 4)(*([0xdeadbeef] * 4))
        address = ctypes.addressof(out)
        call(address, 0xfedcba9876543210, 0)
        expected = [address & 0xffffffff, address >> 32, 0x76543210, 0xfedcba98]
        assert list(out) == expected[:words] + [0xdeadbeef] * (4 - words)
