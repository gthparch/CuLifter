import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("sm, offset", [(75, 0x160), (80, 0x160), (90, 0x210)])
def test_host_wrapper_exposes_kernel_arg_base(sm, offset):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")

    headers = Path(__file__).parent / "common"
    source = (
        f'#include "kernel_wrapper_{sm}.h"\n'
        f"static_assert(SLIFTER_ARG_BASE == {offset});\n"
        f"static_assert(ARG_BASE == {offset});\n"
    )
    subprocess.run(
        [compiler, "-std=c++20", "-I", str(headers), "-fsyntax-only", "-x", "c++", "-"],
        input=source,
        text=True,
        check=True,
    )
