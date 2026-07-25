"""
Test suite for SM 90 architecture (ARM / aarch64).

Reuses the x86 lifter IR but compiles for aarch64 via llvmlite + g++.
Output artifacts use the 'arm' prefix to avoid overwriting x86 results.
"""
import pytest

from .test_utils import (
    execute_test_case,
    get_bases_for_sm_suite,
    get_test_bases,
)

SM = "90"

BASES = get_test_bases()


def _parametrize_suite(suite: str):
    return pytest.mark.parametrize(
        "base",
        get_bases_for_sm_suite(BASES, SM, suite),
        ids=lambda b: b.split("/", 1)[1],
    )


@_parametrize_suite("cuobjdump")
def test_cuobjdump_arm(base, tmp_path):
    execute_test_case(BASES, base, SM, "arm")


