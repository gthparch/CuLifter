"""
Test suite for SM 80 architecture (x86).
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


@_parametrize_suite("cuobjdump")
def test_cuobjdump_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("cuda_handwritten")
def test_cuda_handwritten_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")
