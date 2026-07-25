import argparse
import pathlib
from parse.parser import SaSSParser
from sir.module import Module
from lift.lifter_nvvm import NVVMLifter
from lift.lifter_cpu import CpuLifter
from transform.toggle_reuse import ToggleReuse
from utils.asm import Asm
from utils.elf import Elf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        dest="input_asm",
        required=True,
        nargs="+",
        metavar="ASM",
        help="SASS+ELF or nvdisasm.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output LLVM module name",
        dest="output_module",
        required=True,
    )
    parser.add_argument(
        "-name",
        "--kernel-name",
        help="kernel name",
        dest="kernel_name",
        default="kern",
        type=str,
    )
    parser.add_argument(
        "-arch", dest="arch", default=75, type=int, choices=[70, 75, 80, 86]
    )
    parser.add_argument(
        "--lifter",
        choices=["nvvm", "x86", "arm", "x86warp"],
        default="nvvm",
        help="select lifter backend (x86/arm/x86warp share the CpuLifter; "
             "x86warp = serialized-warp model for warp-specialized SM90 kernels "
             "[fibers + per-warp uniform-register sharing])",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="emit lifter stdout logging"
    )
    parser.add_argument(
        "--emit-sass",
        action="store_true",
        help="emit parsed SASS to output file only (no lifting)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="skip any function that raises exception for best-effort multi-kernel lift",
    )
    parser.add_argument(
        "--const-mem-linkage",
        choices=["external", "internal"],
        default="external",
        help="linkage for the @const_mem global.",
    )
    args = parser.parse_args()

    input_paths = args.input_asm
    if len(input_paths) == 1:
        asm_path = pathlib.Path(input_paths[0])
        asm_text = asm_path.read_text()
        asm = Asm(asm_text, path=asm_path)
        elf = asm
        file = asm_text
    elif len(input_paths) == 2:
        sass_path, elf_path = input_paths
        elf = Elf(elf_path)
        file = pathlib.Path(sass_path).read_text()
        asm = None
    else:
        raise ValueError("Expected SASS+ELF or nvdisasm.txt")

    # nvvm needs desc_b's high word live for full wgmma descriptors;
    # x86warp must not shift its SSA numbering (see sir/instruction.py).
    from sir.instruction import Instruction as _Inst
    _Inst.SYNTH_DESC_B_HI = (args.lifter == "nvvm")

    sass_parser = SaSSParser(file, elf=elf)
    module = Module(args.output_module, sass_parser)

    if args.kernel_name and args.kernel_name != "kern":
        wanted = set(n.strip() for n in args.kernel_name.split(",") if n.strip())
        matches = [f for f in module.functions if f.name in wanted]
        if not matches:
            raise SystemExit(
                f"--kernel-name '{args.kernel_name}' not found in module "
                f"({len(module.functions)} functions parsed)"
            )
        module.functions = matches

    if args.emit_sass:
        ToggleReuse().apply(module)
        file_name = args.output_module + ".sass"
        with open(file_name, "w") as output_file:
            print(module.dump(), file=output_file)
        return

    verbose_mode = bool(args.verbose)
    coe = bool(args.continue_on_error)
    cm_linkage = args.const_mem_linkage
    if args.lifter == "nvvm":
        lifter = NVVMLifter(elf=elf, verbose=verbose_mode,
                            continue_on_error=coe,
                            const_mem_linkage=cm_linkage)
    elif args.lifter in ("x86", "arm", "x86warp"):
        lifter = CpuLifter(elf=elf, verbose=verbose_mode,
                           continue_on_error=coe,
                           const_mem_linkage=cm_linkage)
        # Serialized-warp model: one OS thread per warp, 32 lanes as cooperative
        # fibers. Toggles the warp-mode lift (R2UR -> per-warp shared slot,
        # per-lane const/local, yielding collectives). Off => x86/arm unchanged.
        lifter.warp_mode = (args.lifter == "x86warp")
    else:
        raise ValueError(f"Unknown lifter backend '{args.lifter}'")

    file_name = args.output_module + ".ll"
    with open(file_name, "w") as output_file:
        lifter.lift_module(module, output_file)


if __name__ == "__main__":
    main()
