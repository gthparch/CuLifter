#!/usr/bin/env python3
"""Compile LLVM IR (.ll) to aarch64 object file (.o) using llvmlite.

Usage: python compile_ll_arm.py input.ll output.o

This avoids the need for clang on aarch64 machines.  The .ll produced by
SLifter's x86 lifter uses target triple "unknown-unknown-unknown" and
thread_local globals — both are patched before compilation.
"""
import sys
from pathlib import Path

from llvmlite import binding as llvm


def compile_ll_to_arm(ll_path: str, obj_path: str) -> None:
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    ll_text = Path(ll_path).read_text()

    # Retarget for aarch64
    ll_text = ll_text.replace(
        'target triple = "unknown-unknown-unknown"',
        'target triple = "aarch64-unknown-linux-gnu"',
    )
    # aarch64 ELF TLS requires localexec model
    ll_text = ll_text.replace(
        "external thread_local global",
        "external thread_local(localexec) global",
    )

    mod = llvm.parse_assembly(ll_text)
    mod.verify()

    target = llvm.Target.from_triple("aarch64-unknown-linux-gnu")
    tm = target.create_target_machine(opt=2, codemodel="small")
    obj_bytes = tm.emit_object(mod)
    Path(obj_path).write_bytes(obj_bytes)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} input.ll output.o", file=sys.stderr)
        sys.exit(1)
    compile_ll_to_arm(sys.argv[1], sys.argv[2])
