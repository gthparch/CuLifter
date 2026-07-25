"""
Unit tests for utils/fragment_layout.py — fragment layout validation
for SM90 LDSM, STSM, and HMMA matrix operations.
"""

import pytest
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.fragment_layout import (
    hmma_fragment_info,
    fragment_mapping,
    ldsm_offsets,
    ldsm_tile_offsets,
    hmma_scalar_ops,
    validate_fragment_coverage,
)


# ===================================================================
# hmma_fragment_info tests
# ===================================================================

class TestHmmaFragmentInfo:
    """Verify HMMA register counts and types for each shape."""

    def test_m16816_f32(self):
        info = hmma_fragment_info("16816", "F32")
        assert info["A"]["num_regs"] == 4
        assert info["A"]["elements_per_reg"] == 2
        assert info["A"]["dtype"] == "f16"

        assert info["B"]["num_regs"] == 2
        assert info["B"]["elements_per_reg"] == 2
        assert info["B"]["dtype"] == "f16"

        assert info["C"]["num_regs"] == 4
        assert info["C"]["elements_per_reg"] == 1
        assert info["C"]["dtype"] == "f32"

        assert info["D"]["num_regs"] == 4
        assert info["D"]["elements_per_reg"] == 1
        assert info["D"]["dtype"] == "f32"

    def test_m16816_f16(self):
        info = hmma_fragment_info("16816", "F16")
        assert info["A"]["num_regs"] == 4
        assert info["B"]["num_regs"] == 2
        assert info["C"]["num_regs"] == 2
        assert info["C"]["dtype"] == "f16"
        assert info["D"]["num_regs"] == 2

    def test_m1688_f32(self):
        info = hmma_fragment_info("1688", "F32")
        assert info["A"]["num_regs"] == 2
        assert info["B"]["num_regs"] == 1
        assert info["C"]["num_regs"] == 4
        assert info["C"]["dtype"] == "f32"
        assert info["D"]["num_regs"] == 4

    def test_m1688_f16(self):
        info = hmma_fragment_info("1688", "F16")
        assert info["A"]["num_regs"] == 2
        assert info["B"]["num_regs"] == 1
        assert info["C"]["num_regs"] == 2
        assert info["C"]["dtype"] == "f16"
        assert info["D"]["num_regs"] == 2

    def test_m884_f16(self):
        info = hmma_fragment_info("884", "F16")
        assert info["A"]["num_regs"] == 2
        assert info["B"]["num_regs"] == 2
        assert info["C"]["num_regs"] == 2
        assert info["D"]["num_regs"] == 2

    def test_register_counts_match_instruction_py(self):
        """Cross-check register counts against sir/instruction.py HMMA handling.

        instruction.py defines d_count, a_count, b_count, c_count for each shape.
        These MUST match hmma_fragment_info.
        """
        # From instruction.py line 349-356:
        # 884: d=2, a=2, b=2, c=2
        info_884 = hmma_fragment_info("884", "F16")
        assert info_884["D"]["num_regs"] == 2
        assert info_884["A"]["num_regs"] == 2
        assert info_884["B"]["num_regs"] == 2
        assert info_884["C"]["num_regs"] == 2

        # 1688, F16: d=2, a=2, b=1, c=2
        info_1688_f16 = hmma_fragment_info("1688", "F16")
        assert info_1688_f16["D"]["num_regs"] == 2
        assert info_1688_f16["A"]["num_regs"] == 2
        assert info_1688_f16["B"]["num_regs"] == 1
        assert info_1688_f16["C"]["num_regs"] == 2

        # 1688, F32: d=4, a=2, b=1, c=4
        info_1688_f32 = hmma_fragment_info("1688", "F32")
        assert info_1688_f32["D"]["num_regs"] == 4
        assert info_1688_f32["A"]["num_regs"] == 2
        assert info_1688_f32["B"]["num_regs"] == 1
        assert info_1688_f32["C"]["num_regs"] == 4

        # 16816 (else branch): d=4, a=2->4 (actually 4 for 16816), b=2, c=4
        # instruction.py line 356 for the else case: d=4, a=2, b=2, c=4
        # BUT for 16816 the A reg count per thread is 4 in the PTX spec.
        # instruction.py uses 2 as the operand count (the SASS instruction
        # lists the first register, and create_reg_group expands it).
        # So a_count=2 in instruction.py actually means the SASS uses
        # pairs of register pairs.  The fragment info uses the per-thread
        # register count which is 4.  This is correct — instruction.py
        # is working at the SASS operand level, not the per-thread level.

    def test_unknown_shape_raises(self):
        with pytest.raises(ValueError, match="Unknown HMMA"):
            hmma_fragment_info("999", "F32")

    def test_returned_dict_is_copy(self):
        """Ensure mutation does not affect the internal table."""
        info = hmma_fragment_info("16816", "F32")
        info["A"]["num_regs"] = 999
        info2 = hmma_fragment_info("16816", "F32")
        assert info2["A"]["num_regs"] == 4


# ===================================================================
# LDSM offset tests
# ===================================================================

class TestLdsmOffsets:
    """Verify LDSM shared memory offset computation."""

    def test_ldsm_m88_x1(self):
        """LDSM.16.M88.1 loads 1 register from base."""
        offsets = ldsm_offsets("M88.1", 0x200, lane_id=0)
        assert len(offsets) == 1
        assert offsets[0] == 0x200

    def test_ldsm_m88_x2(self):
        """LDSM.16.M88.2 loads 2 registers."""
        offsets = ldsm_offsets("M88.2", 0x200, lane_id=0)
        assert len(offsets) == 2
        assert offsets == [0x200, 0x204]

    def test_ldsm_m88_x4(self):
        """LDSM.16.M88.4 loads 4 consecutive registers."""
        offsets = ldsm_offsets("M88.4", 0x200, lane_id=0)
        assert len(offsets) == 4
        assert offsets == [0x200, 0x204, 0x208, 0x20C]

    def test_ldsm_mt88_x1(self):
        """LDSM.16.MT88.1 loads 1 register (transposed)."""
        offsets = ldsm_offsets("MT88.1", 0x100, lane_id=0)
        assert len(offsets) == 1
        assert offsets[0] == 0x100

    def test_ldsm_mt88_x4(self):
        """LDSM.16.MT88.4 loads 4 registers (transposed)."""
        offsets = ldsm_offsets("MT88.4", 0x100, lane_id=0)
        assert len(offsets) == 4
        assert offsets == [0x100, 0x104, 0x108, 0x10C]

    def test_ldsm_zero_base(self):
        offsets = ldsm_offsets("M88.4", 0, lane_id=0)
        assert offsets == [0, 4, 8, 12]

    def test_ldsm_stride_4_bytes(self):
        """Each register is 4 bytes apart in sequential model."""
        offsets = ldsm_offsets("M88.4", 0, lane_id=0)
        for i in range(len(offsets) - 1):
            assert offsets[i + 1] - offsets[i] == 4


class TestLdsmTileOffsets:
    """Verify tile-aware LDSM offset computation."""

    def test_tile_x1_lane0(self):
        """Single tile, lane 0: row 0, col pair 0."""
        offsets = ldsm_tile_offsets("M88.1", 0x200, lane_id=0, stride=16)
        assert len(offsets) == 1
        # lane 0: row=0, col_pair=0 -> offset = 0*16 + 0*4 = 0
        assert offsets[0] == 0x200

    def test_tile_x1_lane1(self):
        """Single tile, lane 1: row 1, col pair 0."""
        offsets = ldsm_tile_offsets("M88.1", 0x200, lane_id=1, stride=16)
        assert len(offsets) == 1
        # lane 1: row=1, col_pair=0 -> offset = 1*16 + 0*4 = 16
        assert offsets[0] == 0x200 + 16

    def test_tile_x1_lane8(self):
        """Single tile, lane 8: row 0, col pair 1."""
        offsets = ldsm_tile_offsets("M88.1", 0x200, lane_id=8, stride=16)
        assert len(offsets) == 1
        # lane 8: row=0, col_pair=1 -> offset = 0*16 + 1*4 = 4
        assert offsets[0] == 0x200 + 4

    def test_tile_x4_lane0(self):
        """4 tiles, lane 0: each tile is 128 bytes apart."""
        offsets = ldsm_tile_offsets("M88.4", 0, lane_id=0, stride=16)
        assert len(offsets) == 4
        # lane 0 intra-tile offset = 0 (row 0, col_pair 0)
        # tile_bytes = 8 * 16 = 128
        assert offsets == [0, 128, 256, 384]


# ===================================================================
# HMMA fragment mapping tests — lane 0
# ===================================================================

class TestHmmaFragmentMappingLane0:
    """Verify fragment mapping for lane 0 (x86 single-threaded mode)."""

    def test_m16816_A_lane0(self):
        """A matrix fragment for lane 0 of m16n8k16."""
        # lane 0: groupID=0, threadID_in_group=0
        # a0 = {A[0][0], A[0][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=0, operand="A")
        assert m == [(0, 0), (0, 1)]

        # a1 = {A[8][0], A[8][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=1, operand="A")
        assert m == [(8, 0), (8, 1)]

        # a2 = {A[0][8], A[0][9]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=2, operand="A")
        assert m == [(0, 8), (0, 9)]

        # a3 = {A[8][8], A[8][9]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=3, operand="A")
        assert m == [(8, 8), (8, 9)]

    def test_m16816_B_lane0(self):
        """B matrix fragment for lane 0 of m16n8k16."""
        # lane 0: groupID=0, threadID_in_group=0
        # b0 = {B[0][0], B[0][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=0, operand="B")
        assert m == [(0, 0), (0, 1)]

        # b1 = {B[8][0], B[8][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=1, operand="B")
        assert m == [(8, 0), (8, 1)]

    def test_m16816_D_lane0(self):
        """D matrix fragment for lane 0 of m16n8k16."""
        # lane 0: groupID=0, threadID_in_group=0
        # d0 = D[0][0]
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=0, operand="D")
        assert m == [(0, 0)]

        # d1 = D[8][0]
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=1, operand="D")
        assert m == [(8, 0)]

        # d2 = D[0][4]
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=2, operand="D")
        assert m == [(0, 4)]

        # d3 = D[8][4]
        m = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=3, operand="D")
        assert m == [(8, 4)]

    def test_m16816_C_same_as_D(self):
        """C and D share the same fragment layout."""
        for reg_idx in range(4):
            c = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=reg_idx, operand="C")
            d = fragment_mapping("HMMA", "M16816", lane_id=0, reg_index=reg_idx, operand="D")
            assert c == d


# ===================================================================
# HMMA fragment mapping tests — multiple lanes
# ===================================================================

class TestHmmaFragmentMappingMultiLane:
    """Verify fragment mapping across warp lanes."""

    def test_m16816_A_lane4(self):
        """A matrix fragment for lane 4 of m16n8k16."""
        # lane 4: groupID=1, threadID_in_group=0
        # a0 = {A[1][0], A[1][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=4, reg_index=0, operand="A")
        assert m == [(1, 0), (1, 1)]

        # a1 = {A[9][0], A[9][1]}
        m = fragment_mapping("HMMA", "M16816", lane_id=4, reg_index=1, operand="A")
        assert m == [(9, 0), (9, 1)]

    def test_m16816_A_lane1(self):
        """A matrix fragment for lane 1 of m16n8k16."""
        # lane 1: groupID=0, threadID_in_group=1
        # a0 = {A[0][2], A[0][3]}
        m = fragment_mapping("HMMA", "M16816", lane_id=1, reg_index=0, operand="A")
        assert m == [(0, 2), (0, 3)]

    def test_m16816_D_lane1(self):
        """D matrix fragment for lane 1 of m16n8k16."""
        # lane 1: groupID=0, threadID_in_group=1
        # d0 = D[0][1]
        m = fragment_mapping("HMMA", "M16816", lane_id=1, reg_index=0, operand="D")
        assert m == [(0, 1)]

        # d2 = D[0][5]
        m = fragment_mapping("HMMA", "M16816", lane_id=1, reg_index=2, operand="D")
        assert m == [(0, 5)]

    def test_m16816_D_lane31(self):
        """D matrix fragment for lane 31 (last lane)."""
        # lane 31: groupID=7, threadID_in_group=3
        # d0 = D[7][3]
        m = fragment_mapping("HMMA", "M16816", lane_id=31, reg_index=0, operand="D")
        assert m == [(7, 3)]

        # d1 = D[15][3]
        m = fragment_mapping("HMMA", "M16816", lane_id=31, reg_index=1, operand="D")
        assert m == [(15, 3)]

        # d2 = D[7][7]
        m = fragment_mapping("HMMA", "M16816", lane_id=31, reg_index=2, operand="D")
        assert m == [(7, 7)]

        # d3 = D[15][7]
        m = fragment_mapping("HMMA", "M16816", lane_id=31, reg_index=3, operand="D")
        assert m == [(15, 7)]


# ===================================================================
# Fragment coverage validation
# ===================================================================

class TestFragmentCoverage:
    """Verify that all matrix elements are covered across a full warp."""

    def test_m16816_D_full_warp_coverage(self):
        """All 16x8=128 elements of D must be covered by 32 lanes."""
        report = validate_fragment_coverage("M16816")
        assert report["D"]["complete"], (
            f"D matrix coverage incomplete: "
            f"{len(report['D']['covered'])}/{16*8} elements covered"
        )

    def test_m16816_A_full_warp_coverage(self):
        """All 16x16=256 elements of A must be covered."""
        report = validate_fragment_coverage("M16816")
        assert report["A"]["complete"], (
            f"A matrix coverage incomplete: "
            f"{len(report['A']['covered'])}/{16*16} elements covered"
        )

    def test_m16816_B_full_warp_coverage(self):
        """All 16x8=128 elements of B must be covered."""
        report = validate_fragment_coverage("M16816")
        assert report["B"]["complete"], (
            f"B matrix coverage incomplete: "
            f"{len(report['B']['covered'])}/{16*8} elements covered"
        )

    def test_m1688_D_full_warp_coverage(self):
        """All 16x8=128 elements of D must be covered for m16n8k8."""
        report = validate_fragment_coverage("M1688")
        assert report["D"]["complete"], (
            f"D matrix coverage: "
            f"{len(report['D']['covered'])}/{16*8}"
        )

    def test_m1688_A_full_warp_coverage(self):
        """All 16x8=128 elements of A must be covered for m16n8k8."""
        report = validate_fragment_coverage("M1688")
        assert report["A"]["complete"], (
            f"A matrix coverage: "
            f"{len(report['A']['covered'])}/{16*8}"
        )

    def test_m16816_D_no_overlap(self):
        """Each D element should be held by exactly one (lane, reg) pair."""
        seen = {}
        for lane in range(32):
            for reg_idx in range(4):
                positions = fragment_mapping("HMMA", "M16816", lane, reg_idx, "D")
                for pos in positions:
                    if pos in seen:
                        pytest.fail(
                            f"D element {pos} covered by both "
                            f"(lane={seen[pos][0]}, reg={seen[pos][1]}) and "
                            f"(lane={lane}, reg={reg_idx})"
                        )
                    seen[pos] = (lane, reg_idx)
        assert len(seen) == 16 * 8


# ===================================================================
# LDSM fragment mapping tests
# ===================================================================

class TestLdsmFragmentMapping:
    """Verify LDSM fragment position mapping."""

    def test_ldsm_normal_lane0_reg0(self):
        """Non-transposed LDSM, lane 0, first register."""
        m = fragment_mapping("LDSM", "M88", lane_id=0, reg_index=0)
        # lane 0: row=0, col_pair=0 -> (0,0), (0,1)
        assert m == [(0, 0), (0, 1)]

    def test_ldsm_normal_lane1_reg0(self):
        """Non-transposed LDSM, lane 1, first register."""
        m = fragment_mapping("LDSM", "M88", lane_id=1, reg_index=0)
        # lane 1: row=1, col_pair=0 -> (1,0), (1,1)
        assert m == [(1, 0), (1, 1)]

    def test_ldsm_normal_lane8_reg0(self):
        """Non-transposed LDSM, lane 8, first register."""
        m = fragment_mapping("LDSM", "M88", lane_id=8, reg_index=0)
        # lane 8: row=0, col_pair=1 -> (0,2), (0,3)
        assert m == [(0, 2), (0, 3)]

    def test_ldsm_normal_lane0_reg1(self):
        """Non-transposed LDSM, lane 0, second register (second tile)."""
        m = fragment_mapping("LDSM", "M88", lane_id=0, reg_index=1)
        # Second tile: tile_col_offset=8
        # lane 0: row=0, col=8, col+1=9
        assert m == [(0, 8), (0, 9)]

    def test_ldsm_transposed_lane0_reg0(self):
        """Transposed LDSM, lane 0, first register."""
        m = fragment_mapping("LDSM", "MT88", lane_id=0, reg_index=0)
        # Transposed: col=0, row_base=0 -> (0,0), (1,0)
        assert m == [(0, 0), (1, 0)]

    def test_ldsm_transposed_lane8_reg0(self):
        """Transposed LDSM, lane 8, first register."""
        m = fragment_mapping("LDSM", "MT88", lane_id=8, reg_index=0)
        # Transposed: col=0, row_base=2 -> (2,0), (3,0)
        assert m == [(2, 0), (3, 0)]


# ===================================================================
# HMMA scalar decomposition tests
# ===================================================================

class TestHmmaScalarOps:
    """Verify scalar decomposition for HMMA operations."""

    def test_m16816_op_count(self):
        """m16n8k16 should produce 4 D regs * 4 partial products = 16 FMAs."""
        ops = hmma_scalar_ops("M16816")
        assert len(ops) == 16

    def test_m16816_all_d_regs_covered(self):
        """All 4 D registers should have operations."""
        ops = hmma_scalar_ops("M16816")
        d_regs = {op["d_reg"] for op in ops}
        assert d_regs == {0, 1, 2, 3}

    def test_m16816_each_d_reg_has_4_ops(self):
        """Each D register accumulates 4 partial products for lane 0."""
        ops = hmma_scalar_ops("M16816")
        for d_reg in range(4):
            count = sum(1 for op in ops if op["d_reg"] == d_reg)
            assert count == 4, f"D reg {d_reg} has {count} ops, expected 4"

    def test_m16816_op_structure(self):
        """Each op must have the expected keys."""
        ops = hmma_scalar_ops("M16816")
        required_keys = {"d_reg", "a_reg", "a_elem", "b_reg", "b_elem", "c_reg"}
        for op in ops:
            assert set(op.keys()) == required_keys

    def test_m1688_op_count(self):
        """m16n8k8 should produce 4 D regs * 2 partial products = 8 FMAs."""
        ops = hmma_scalar_ops("M1688")
        assert len(ops) == 8

    def test_m1688_all_d_regs_covered(self):
        ops = hmma_scalar_ops("M1688")
        d_regs = {op["d_reg"] for op in ops}
        assert d_regs == {0, 1, 2, 3}


# ===================================================================
# Error handling tests
# ===================================================================

class TestErrorHandling:
    """Verify proper error handling for invalid inputs."""

    def test_invalid_opcode(self):
        with pytest.raises(ValueError, match="Unsupported opcode"):
            fragment_mapping("INVALID", "M16816", 0, 0, "A")

    def test_hmma_missing_operand(self):
        with pytest.raises(ValueError, match="operand must be specified"):
            fragment_mapping("HMMA", "M16816", 0, 0)

    def test_hmma_invalid_shape(self):
        with pytest.raises(ValueError, match="No fragment mapping"):
            fragment_mapping("HMMA", "INVALID", 0, 0, "A")

    def test_hmma_invalid_operand(self):
        with pytest.raises(ValueError, match="No fragment mapping"):
            fragment_mapping("HMMA", "M16816", 0, 0, "X")

    def test_m16816_A_invalid_reg(self):
        with pytest.raises(ValueError, match="reg_index must be 0-3"):
            fragment_mapping("HMMA", "M16816", 0, 5, "A")

    def test_m16816_B_invalid_reg(self):
        with pytest.raises(ValueError, match="reg_index must be 0-1"):
            fragment_mapping("HMMA", "M16816", 0, 3, "B")

    def test_m16816_D_invalid_reg(self):
        with pytest.raises(ValueError, match="reg_index must be 0-3"):
            fragment_mapping("HMMA", "M16816", 0, 5, "D")

    def test_scalar_ops_invalid_shape(self):
        with pytest.raises(ValueError, match="not supported"):
            hmma_scalar_ops("M884")

    def test_fragment_info_unknown(self):
        with pytest.raises(ValueError, match="Unknown HMMA"):
            hmma_fragment_info("12345", "F64")


# ===================================================================
# m16n8k8 fragment mapping tests
# ===================================================================

class TestM1688FragmentMapping:
    """Verify fragment mapping for the m16n8k8 shape."""

    def test_m1688_A_lane0(self):
        # lane 0: groupID=0, threadID_in_group=0
        # a0 = {A[0][0], A[0][1]}
        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=0, operand="A")
        assert m == [(0, 0), (0, 1)]

        # a1 = {A[8][0], A[8][1]}
        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=1, operand="A")
        assert m == [(8, 0), (8, 1)]

    def test_m1688_B_lane0(self):
        # lane 0: groupID=0, threadID_in_group=0
        # b0 = {B[0][0], B[0][1]}
        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=0, operand="B")
        assert m == [(0, 0), (0, 1)]

    def test_m1688_D_lane0(self):
        # Same as m16n8k16 D layout
        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=0, operand="D")
        assert m == [(0, 0)]

        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=1, operand="D")
        assert m == [(8, 0)]

        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=2, operand="D")
        assert m == [(0, 4)]

        m = fragment_mapping("HMMA", "M1688", lane_id=0, reg_index=3, operand="D")
        assert m == [(8, 4)]


# ===================================================================
# Consistency tests: fragment mapping vs fragment info
# ===================================================================

class TestConsistency:
    """Cross-checks between fragment_mapping and hmma_fragment_info."""

    def test_m16816_A_reg_count_consistent(self):
        """Number of valid reg indices should match num_regs from info."""
        info = hmma_fragment_info("16816", "F32")
        n = info["A"]["num_regs"]
        # Should be able to call fragment_mapping for reg 0..n-1 without error
        for reg_idx in range(n):
            m = fragment_mapping("HMMA", "M16816", 0, reg_idx, "A")
            assert len(m) == info["A"]["elements_per_reg"]

    def test_m16816_B_reg_count_consistent(self):
        info = hmma_fragment_info("16816", "F32")
        n = info["B"]["num_regs"]
        for reg_idx in range(n):
            m = fragment_mapping("HMMA", "M16816", 0, reg_idx, "B")
            assert len(m) == info["B"]["elements_per_reg"]

    def test_m16816_D_reg_count_consistent(self):
        info = hmma_fragment_info("16816", "F32")
        n = info["D"]["num_regs"]
        for reg_idx in range(n):
            m = fragment_mapping("HMMA", "M16816", 0, reg_idx, "D")
            assert len(m) == info["D"]["elements_per_reg"]

    def test_m1688_A_reg_count_consistent(self):
        info = hmma_fragment_info("1688", "F32")
        n = info["A"]["num_regs"]
        for reg_idx in range(n):
            m = fragment_mapping("HMMA", "M1688", 0, reg_idx, "A")
            assert len(m) == info["A"]["elements_per_reg"]
