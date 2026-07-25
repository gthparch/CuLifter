"""
Test suite for SM 75 architecture (NVVM).
"""
import pytest

from .test_utils import (
    execute_nvdisasm_emit_case,
    execute_test_case,
    get_bases_for_sm_suite,
    get_test_bases,
)

SM = "75"

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


@_parametrize_suite("nvdisasm")
def test_nvdisasm_emit(base, tmp_path):
    execute_nvdisasm_emit_case(BASES, base, SM)


@_parametrize_suite("ml")
def test_ml_nvvm(base, tmp_path):
    execute_test_case(BASES, base, SM, "nvvm")


