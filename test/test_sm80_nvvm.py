"""
Test suite for SM 80 architecture (NVVM).

Mirrors test_sm75_nvvm.py / test_sm90_nvvm.py but with SM="80" — used
for HeCBench tests whose source requires CUDA bfloat / FP8 APIs that
are guarded on `__CUDA_ARCH__ >= 800` (e.g. addBiasQKV, gelu, f16atomic).
The lifter handles sm_80 SASS via SaSSParser_SM75 (per parse/parser.py
dispatch); BF16x2 lifter paths emit native `<2 x bfloat>` IR (LLVM 20+
NVPTX has native bfloat selection).
"""
import pytest

from .test_utils import (
    execute_test_case,
    get_bases_for_sm_suite,
    get_test_bases,
)

SM = "80"

# Initialize test bases (identification based on cu or sass presence)
BASES = get_test_bases()


def _parametrize_suite(suite: str):
    bases = get_bases_for_sm_suite(BASES, SM, suite)
    if not bases:
        bases = [pytest.param("__none__", marks=pytest.mark.skip(reason=f"no {suite} bases for SM{SM}"))]
    return pytest.mark.parametrize(
        "base",
        bases,
        ids=lambda b: b.split("/", 1)[1] if isinstance(b, str) and "/" in b else str(b),
    )


@_parametrize_suite("nvbit")
def test_nvbit_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")


@_parametrize_suite("cuobjdump")
def test_cuobjdump_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")


@_parametrize_suite("nvdisasm")
def test_nvdisasm_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")


