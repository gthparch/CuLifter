"""
Test suite for SM 90 architecture (x86).
"""
import pytest

from .test_utils import (
    execute_test_case,
    get_bases_for_sm_suite,
    get_test_bases,
)

SM = "90"

# Initialize test bases (identification based on cu or sass presence)
BASES = get_test_bases()


def _parametrize_suite(suite: str):
    return pytest.mark.parametrize(
        "base",
        get_bases_for_sm_suite(BASES, SM, suite),
        ids=lambda b: b.split("/", 1)[1],
    )


@_parametrize_suite("nvbit")
def test_nvbit_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("cuobjdump")
def test_cuobjdump_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("addrtest")
def test_addrtest_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")


@_parametrize_suite("cuda_handwritten")
def test_cuda_handwritten_x86(base, tmp_path):
    execute_test_case(BASES, base, SM, "x86")
