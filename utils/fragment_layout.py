"""
Fragment layout validation for SM90 matrix operations (LDSM, STSM, HMMA).

Encodes the NVIDIA PTX ISA register-to-matrix-element mappings used by
ldmatrix (LDSM/STSM) and mma.sync (HMMA) instructions.  Current consumer:
test/test_fragment_layout.py (layout-table validation); the x86 lifter
implements its HMMA/WGMMA layouts independently in lift/lifter_cpu.py.

References:
  - NVIDIA PTX ISA 8.x, Section 9.7.14: Matrix Multiply-Accumulate
  - NVIDIA PTX ISA 8.x, Section 9.7.13: ldmatrix
  - CUDA Programming Guide: Warp Matrix Functions

Layout tables are encoded for the shapes used in SM75 (Volta/Turing) and
SM90 (Hopper):

  HMMA.884      — mma.sync.aligned.m8n8k4.f16.f16
  HMMA.1688.F16 — mma.sync.aligned.m16n8k8.f16.f16
  HMMA.1688.F32 — mma.sync.aligned.m16n8k8.f16.f32
  HMMA.16816.F32— mma.sync.aligned.m16n8k16.f16.f32 (SM90 primary shape)

  LDSM.16.M88   — ldmatrix.sync.aligned.m8n8 (.x1/.x2/.x4)
  LDSM.16.MT88  — ldmatrix.sync.aligned.m8n8.trans (.x1/.x2/.x4)
"""


# ---------------------------------------------------------------------------
# HMMA fragment register info
# ---------------------------------------------------------------------------

# Shape key → operand info dict
# Each operand (A, B, C, D) stores:
#   num_regs       — number of 32-bit registers per thread
#   elements_per_reg — number of data elements packed per register
#   dtype          — element data type string ("f16", "f32", "i32", etc.)
_HMMA_FRAGMENT_INFO = {
    # mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16
    ("884", "F16"): {
        "A": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "B": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "C": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "D": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
    },
    # mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16
    ("1688", "F16"): {
        "A": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "B": {"num_regs": 1, "elements_per_reg": 2, "dtype": "f16"},
        "C": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "D": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
    },
    # mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    ("1688", "F32"): {
        "A": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "B": {"num_regs": 1, "elements_per_reg": 2, "dtype": "f16"},
        "C": {"num_regs": 4, "elements_per_reg": 1, "dtype": "f32"},
        "D": {"num_regs": 4, "elements_per_reg": 1, "dtype": "f32"},
    },
    # mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    ("16816", "F32"): {
        "A": {"num_regs": 4, "elements_per_reg": 2, "dtype": "f16"},
        "B": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "C": {"num_regs": 4, "elements_per_reg": 1, "dtype": "f32"},
        "D": {"num_regs": 4, "elements_per_reg": 1, "dtype": "f32"},
    },
    # mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    ("16816", "F16"): {
        "A": {"num_regs": 4, "elements_per_reg": 2, "dtype": "f16"},
        "B": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "C": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
        "D": {"num_regs": 2, "elements_per_reg": 2, "dtype": "f16"},
    },
}

# ---------------------------------------------------------------------------
# WGMMA/HGMMA fragment register info (SM90 warpgroup MMA)
# ---------------------------------------------------------------------------
# WGMMA operates on a warp group (4 warps, M=64).  Per-thread register counts
# for warp 0 (the only one emulated on x86) match the HMMA m16n8k16 layout
# since each warp within the warpgroup handles a 16-row slice.
#
# For register-based forms:
#   A: 4 regs (8 FP16 values), same as HMMA.16816
#   B: N/4 regs (each holding 2 FP16), minimum 2
#   C/D: N/2 regs for F32, N/4 for F16
def _make_wgmma_fragment(n_dim, accum):
    """Generate WGMMA fragment info for a given N dimension and accumulator type.

    A fragment is always 4 regs (per-warp m16, 8 FP16 values).
    B fragment: N/4 regs, minimum 2 (each holds 2 FP16).
    C/D fragment: N/2 regs for F32 (1 element per reg), N/4 regs for F16 (2 elements per reg).
    """
    b_regs = max(n_dim // 4, 2)
    if accum == "F32":
        cd_regs = max(n_dim // 2, 4)
        return {
            "A": {"num_regs": 4, "elements_per_reg": 2, "dtype": "f16"},
            "B": {"num_regs": b_regs, "elements_per_reg": 2, "dtype": "f16"},
            "C": {"num_regs": cd_regs, "elements_per_reg": 1, "dtype": "f32"},
            "D": {"num_regs": cd_regs, "elements_per_reg": 1, "dtype": "f32"},
        }
    else:  # F16
        cd_regs = max(n_dim // 4, 2)
        return {
            "A": {"num_regs": 4, "elements_per_reg": 2, "dtype": "f16"},
            "B": {"num_regs": b_regs, "elements_per_reg": 2, "dtype": "f16"},
            "C": {"num_regs": cd_regs, "elements_per_reg": 2, "dtype": "f16"},
            "D": {"num_regs": cd_regs, "elements_per_reg": 2, "dtype": "f16"},
        }

_WGMMA_FRAGMENT_INFO = {}
for _n in (8, 16, 32, 64, 128, 256):
    for _acc in ("F32", "F16"):
        _WGMMA_FRAGMENT_INFO[(f"64x{_n}x16", _acc)] = _make_wgmma_fragment(_n, _acc)


def hmma_fragment_info(shape, accum_dtype="F32"):
    """Return register layout info for an HMMA shape.

    Parameters
    ----------
    shape : str
        HMMA shape string, e.g. "884", "1688", "16816".
    accum_dtype : str
        Accumulator data type, e.g. "F32", "F16".

    Returns
    -------
    dict
        Mapping from operand name ("A", "B", "C", "D") to a dict with
        keys "num_regs", "elements_per_reg", "dtype".

    Raises
    ------
    ValueError
        If the (shape, accum_dtype) combination is not recognized.
    """
    key = (shape, accum_dtype.upper())
    if key not in _HMMA_FRAGMENT_INFO:
        raise ValueError(
            f"Unknown HMMA shape/accum: shape={shape!r}, accum={accum_dtype!r}. "
            f"Known: {list(_HMMA_FRAGMENT_INFO.keys())}"
        )
    # Return a deep copy to prevent mutation of the canonical table.
    import copy
    return copy.deepcopy(_HMMA_FRAGMENT_INFO[key])


def wgmma_fragment_info(shape, accum_dtype="F32"):
    """Return register layout info for a WGMMA/HGMMA shape.

    Parameters
    ----------
    shape : str
        WGMMA shape string, e.g. "64x8x16", "64x16x16".
    accum_dtype : str
        Accumulator data type, e.g. "F32", "F16".

    Returns
    -------
    dict
        Mapping from operand name ("A", "B", "C", "D") to a dict with
        keys "num_regs", "elements_per_reg", "dtype".

    Raises
    ------
    ValueError
        If the (shape, accum_dtype) combination is not recognized.
    """
    key = (shape, accum_dtype.upper())
    if key not in _WGMMA_FRAGMENT_INFO:
        raise ValueError(
            f"Unknown WGMMA shape/accum: shape={shape!r}, accum={accum_dtype!r}. "
            f"Known: {list(_WGMMA_FRAGMENT_INFO.keys())}"
        )
    import copy
    return copy.deepcopy(_WGMMA_FRAGMENT_INFO[key])


# ---------------------------------------------------------------------------
# HMMA thread-to-matrix-element fragment mapping
# ---------------------------------------------------------------------------

# PTX ISA mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
# Thread-to-matrix-element mapping for the primary SM90 shape.
#
# Each entry: (lane_id, reg_index, element_in_reg) -> (row, col)
#
# The full layout for all 32 lanes is complex.  For x86 single-threaded
# mode we primarily need lane 0.  We encode the full formula so that any
# lane can be queried.
#
# From the PTX ISA specification for m16n8k16:
#
# ** Operand A (16x16, f16, row-major) **
#   Thread T (lane_id) holds 4 registers {a0, a1, a2, a3}, each with 2 f16:
#     a0 = {A[row0][k0],   A[row0][k0+1]}
#     a1 = {A[row0+8][k0], A[row0+8][k0+1]}
#     a2 = {A[row0][k0+8], A[row0][k0+8+1]}
#     a3 = {A[row0+8][k0+8], A[row0+8][k0+8+1]}
#   where:
#     row0 = (lane_id % 16)          — bottom 4 bits select row within half
#     k0   = (lane_id // 16) * 2     — top bit selects k column pair (unsure)
#   Actually, a more precise reading of the PTX ISA (m16n8k16) gives:
#     groupID   = lane_id >> 2   (0..7)
#     threadID_in_group = lane_id & 3
#     row for a0 = groupID         (0..7)
#     row for a1 = groupID + 8     (8..15)
#     col for a0 = threadID_in_group * 2  (0,2,4,6)
#     col for a2 = threadID_in_group * 2 + 8  (8,10,12,14)
#     a0 has 2 f16: cols [col, col+1]
#
# ** Operand B (16x8, f16, col-major — but shaped as k×n = 16×8) **
#   Thread T holds 2 registers {b0, b1}, each with 2 f16:
#     groupID = lane_id >> 2
#     threadID_in_group = lane_id & 3
#     b0 row (k-dim) = groupID, cols = threadID_in_group * 2, +1
#     b1 row (k-dim) = groupID + 8, cols = threadID_in_group * 2, +1
#
# ** Operand C/D (16x8, f32) **
#   Thread T holds 4 registers {c0, c1, c2, c3}, each with 1 f32:
#     groupID = lane_id >> 2
#     threadID_in_group = lane_id & 3
#     c0 = D[groupID][threadID_in_group]
#     c1 = D[groupID + 8][threadID_in_group]
#     c2 = D[groupID][threadID_in_group + 4]
#     c3 = D[groupID + 8][threadID_in_group + 4]


def _m16n8k16_A_mapping(lane_id, reg_index):
    """Map (lane_id, reg_index) to list of (row, col) for A operand.

    A is 16x16 (M=16, K=16), f16, row-major. Each register holds 2 f16.

    Returns list of (row, col) tuples, one per element in the register.
    """
    groupID = lane_id >> 2
    threadID_in_group = lane_id & 3

    if reg_index == 0:
        row = groupID
        col_base = threadID_in_group * 2
    elif reg_index == 1:
        row = groupID + 8
        col_base = threadID_in_group * 2
    elif reg_index == 2:
        row = groupID
        col_base = threadID_in_group * 2 + 8
    elif reg_index == 3:
        row = groupID + 8
        col_base = threadID_in_group * 2 + 8
    else:
        raise ValueError(f"A operand reg_index must be 0-3, got {reg_index}")

    return [(row, col_base), (row, col_base + 1)]


def _m16n8k16_B_mapping(lane_id, reg_index):
    """Map (lane_id, reg_index) to list of (row, col) for B operand.

    B is 16x8 (K=16, N=8), f16, col-major. Each register holds 2 f16.
    Here 'row' is in the K dimension, 'col' is in the N dimension.

    Returns list of (k_row, n_col) tuples.
    """
    groupID = lane_id >> 2
    threadID_in_group = lane_id & 3

    if reg_index == 0:
        k_row = groupID
        col_base = threadID_in_group * 2
    elif reg_index == 1:
        k_row = groupID + 8
        col_base = threadID_in_group * 2
    else:
        raise ValueError(f"B operand reg_index must be 0-1, got {reg_index}")

    return [(k_row, col_base), (k_row, col_base + 1)]


def _m16n8k16_CD_mapping(lane_id, reg_index):
    """Map (lane_id, reg_index) to (row, col) for C or D operand.

    C/D is 16x8 (M=16, N=8), f32. Each register holds 1 f32.

    Returns list with single (row, col) tuple.
    """
    groupID = lane_id >> 2
    threadID_in_group = lane_id & 3

    if reg_index == 0:
        row = groupID
        col = threadID_in_group
    elif reg_index == 1:
        row = groupID + 8
        col = threadID_in_group
    elif reg_index == 2:
        row = groupID
        col = threadID_in_group + 4
    elif reg_index == 3:
        row = groupID + 8
        col = threadID_in_group + 4
    else:
        raise ValueError(f"C/D operand reg_index must be 0-3, got {reg_index}")

    return [(row, col)]


# m16n8k8 (HMMA.1688) layout — smaller K dimension, otherwise similar structure
def _m16n8k8_A_mapping(lane_id, reg_index):
    """A operand for m16n8k8: 16x8 matrix (M=16, K=8), f16.
    2 registers per thread, each with 2 f16.
    """
    groupID = lane_id >> 2
    threadID_in_group = lane_id & 3

    if reg_index == 0:
        row = groupID
        col_base = threadID_in_group * 2
    elif reg_index == 1:
        row = groupID + 8
        col_base = threadID_in_group * 2
    else:
        raise ValueError(f"A operand reg_index must be 0-1, got {reg_index}")

    return [(row, col_base), (row, col_base + 1)]


def _m16n8k8_B_mapping(lane_id, reg_index):
    """B operand for m16n8k8: 8x8 matrix (K=8, N=8), f16.
    1 register per thread, with 2 f16.
    """
    groupID = lane_id >> 2
    threadID_in_group = lane_id & 3

    if reg_index != 0:
        raise ValueError(f"B operand reg_index must be 0, got {reg_index}")

    k_row = groupID
    col_base = threadID_in_group * 2

    return [(k_row, col_base), (k_row, col_base + 1)]


def _m16n8k8_CD_mapping(lane_id, reg_index):
    """C/D operand for m16n8k8: 16x8 (M=16, N=8), f32 or f16.
    For F32: 4 registers, 1 element each.
    For F16: 2 registers, 2 elements each.
    Layout is same as m16n8k16 C/D for the F32 case.
    """
    return _m16n8k16_CD_mapping(lane_id, reg_index)


# Dispatch table: (shape, operand) -> mapping function
_FRAGMENT_MAPPING = {
    ("M16816", "A"): _m16n8k16_A_mapping,
    ("M16816", "B"): _m16n8k16_B_mapping,
    ("M16816", "C"): _m16n8k16_CD_mapping,
    ("M16816", "D"): _m16n8k16_CD_mapping,
    ("M1688", "A"): _m16n8k8_A_mapping,
    ("M1688", "B"): _m16n8k8_B_mapping,
    ("M1688", "C"): _m16n8k8_CD_mapping,
    ("M1688", "D"): _m16n8k8_CD_mapping,
}


def fragment_mapping(opcode, shape, lane_id, reg_index, operand=None):
    """Map (opcode, shape, lane_id, reg_index) to logical matrix positions.

    Parameters
    ----------
    opcode : str
        "HMMA", "LDSM", "STSM"
    shape : str
        Shape identifier: "M88", "M1688", "M16816"
    lane_id : int
        Thread lane within warp (0-31).
    reg_index : int
        Which register in the fragment (0, 1, 2, ...).
    operand : str or None
        For HMMA: "A", "B", "C", "D".
        For LDSM/STSM: ignored (None).

    Returns
    -------
    list of (int, int)
        List of (row, col) logical matrix positions for each element
        stored in this register for this thread.

    Raises
    ------
    ValueError
        If the combination is not supported.
    """
    if opcode == "HMMA":
        if operand is None:
            raise ValueError("operand must be specified for HMMA")
        key = (shape, operand.upper())
        if key not in _FRAGMENT_MAPPING:
            raise ValueError(
                f"No fragment mapping for HMMA shape={shape!r}, "
                f"operand={operand!r}. Known: {list(_FRAGMENT_MAPPING.keys())}"
            )
        return _FRAGMENT_MAPPING[key](lane_id, reg_index)

    elif opcode in ("LDSM", "STSM"):
        return _ldsm_fragment_mapping(shape, lane_id, reg_index)

    else:
        raise ValueError(f"Unsupported opcode: {opcode!r}")


def _ldsm_fragment_mapping(shape, lane_id, reg_index):
    """Fragment mapping for LDSM/STSM (ldmatrix/stmatrix).

    ldmatrix.sync.aligned.m8n8 loads an 8x8 matrix of 16-bit elements.
    Each thread contributes one row of 8 elements (=16 bytes = 128 bits),
    but the register distribution gives each thread a 32-bit register
    holding 2 consecutive f16 values from a specific row.

    For ldmatrix (non-transposed), the register mapping is:
      Thread T, register R contains row (T % 8), cols based on thread
      and register index.

    For x86 single-threaded (lane 0), we compute the relevant entries.

    Returns list of (row, col) for elements in this register.
    """
    if "MT88" in shape or "MT" in shape:
        # Transposed variant — columns and rows are swapped
        return _ldsm_transposed_mapping(lane_id, reg_index)
    else:
        return _ldsm_normal_mapping(lane_id, reg_index)


def _ldsm_normal_mapping(lane_id, reg_index):
    """ldmatrix.sync.aligned.m8n8 (non-transposed).

    In the non-transposed variant, each group of 8 threads loads one 8x8
    matrix tile.  Thread T within a group of 8 provides the address for
    row (T % 8).  After the warp shuffle, thread T ends up with data
    from a specific (row, col_pair).

    For the standard ldmatrix mapping (PTX ISA):
      Register reg_index for thread lane_id holds:
        row = (lane_id % 8) for the first matrix in the group
              (lane_id % 8) + 8 for the second matrix etc. but
              for a single .x1, each thread's register holds 2 f16 from:
                row = (lane_id % 8)
                cols = reg_index * 8 + (lane_id // 8) * 2, +1

    Simplified for .x1 (single matrix):
      row = lane_id % 8
      col = (lane_id // 8) * 2   (for reg_index=0)
    """
    # For M88 (8x8 tile), each .xN loads N consecutive 8x8 tiles.
    # Within each tile:
    #   row = lane_id % 8
    #   col_pair_idx = lane_id // 8  (0..3, selects which pair of 2 f16)
    # So thread lane_id gets: row = lane_id%8, col = (lane_id//8)*2
    # For .x2/.x4, reg_index selects which 8x8 tile.
    row = lane_id % 8
    col_base = (lane_id // 8) * 2

    # reg_index selects which 8x8 sub-matrix in multi-tile loads
    # For .x4: reg 0,1,2,3 correspond to the four consecutive 8x8 tiles
    # The tiles are laid out in the column direction for row-major:
    # tile 0: cols 0-7, tile 1: cols 8-15, etc.
    # Within each tile, lane mapping is the same.
    tile_col_offset = reg_index * 8

    return [(row, tile_col_offset + col_base),
            (row, tile_col_offset + col_base + 1)]


def _ldsm_transposed_mapping(lane_id, reg_index):
    """ldmatrix.sync.aligned.m8n8.trans (transposed variant).

    In the transposed variant, each thread's register holds 2 f16
    from a column of the matrix instead of a row:
      col = lane_id % 8
      row_base = (lane_id // 8) * 2

    With reg_index selecting the tile.
    """
    col = lane_id % 8
    row_base = (lane_id // 8) * 2

    tile_row_offset = reg_index * 8

    return [(tile_row_offset + row_base, col),
            (tile_row_offset + row_base + 1, col)]


# ---------------------------------------------------------------------------
# LDSM shared memory offsets (for x86 single-threaded execution)
# ---------------------------------------------------------------------------

def ldsm_offsets(variant, base_addr, lane_id=0):
    """Compute shared memory byte offsets for LDSM loads.

    In x86 single-threaded mode (lane 0), LDSM is decomposed into
    sequential 4-byte (32-bit register) loads from shared memory.

    Parameters
    ----------
    variant : str
        LDSM variant: "M88.1", "M88.2", "M88.4",
                       "MT88.1", "MT88.2", "MT88.4"
    base_addr : int
        Base shared memory address (byte offset).
    lane_id : int
        Thread lane (0-31). For x86, always 0.

    Returns
    -------
    list of int
        Byte offsets for each register load.

    Notes
    -----
    On actual GPU hardware, each thread provides its own address and the
    ldmatrix instruction performs a warp-wide cooperative load with
    internal shuffles.  For x86 single-threaded emulation, we need to
    determine what addresses lane 0 would provide.

    In the GPU ldmatrix protocol:
    - Each group of 8 threads provides addresses for 8 rows of an 8x8 tile
    - Thread T in the group provides the address for row (T % 8)
    - Each row is 8 * sizeof(f16) = 16 bytes
    - Lane 0 provides the address for row 0

    For x86 lane 0, the address parameter IS the base address (row 0's
    start).  After the warp shuffle, lane 0 ends up with 32 bits (2 f16)
    from row 0.  So for each register, the offset is computed from the
    base address for that tile.

    For multi-tile loads (.x2, .x4), the tiles are loaded from addresses
    provided by different groups of 8 threads.  In x86 mode, we compute
    sequential offsets assuming the tiles are contiguous: each 8x8 tile
    is 8 rows * 16 bytes/row = 128 bytes.

    However, the simpler model for x86 (matching the lifter's current
    implementation) treats LDSM as sequential 4-byte loads:
    """
    # Parse variant to get count
    parts = variant.split(".")
    count = int(parts[-1]) if parts[-1].isdigit() else 1

    # For x86 single-threaded: sequential 4-byte loads from base
    # This matches the lifter's current decomposition: each register
    # is loaded from base + reg_index * 4
    offsets = []
    for i in range(count):
        offsets.append(base_addr + i * 4)

    return offsets


def ldsm_tile_offsets(variant, base_addr, lane_id=0, stride=16):
    """Compute shared memory offsets with proper tile layout awareness.

    Unlike `ldsm_offsets` which gives the simplified sequential model,
    this function computes offsets based on the actual matrix layout
    in shared memory.

    Parameters
    ----------
    variant : str
        LDSM variant: "M88.1", "M88.2", "M88.4", "MT88.1", etc.
    base_addr : int
        Base shared memory address (byte offset) for row 0 of tile 0.
    lane_id : int
        Thread lane (0-31).
    stride : int
        Row stride in bytes in shared memory (default: 16 for 8 f16 per row).

    Returns
    -------
    list of int
        Byte offsets for each register load, accounting for the matrix
        layout in shared memory.

    Notes
    -----
    For lane 0 with stride=16:
      - .x1: [base_addr + 0]  (row 0, first 4 bytes = 2 f16)
      - .x2: [base_addr + 0, base_addr + 128]  (tile 0 row 0, tile 1 row 0)
      - .x4: [base_addr + 0, base_addr + 128, base_addr + 256, base_addr + 384]

    This is the hardware-accurate model.  The per-register offset for
    lane_id within a single tile is:
      row_in_tile = lane_id % 8  (which row this lane provides/receives)
      col_pair    = lane_id // 8 (which 4-byte chunk within the row)
      offset = row_in_tile * stride + col_pair * 4
    """
    parts = variant.split(".")
    count = int(parts[-1]) if parts[-1].isdigit() else 1
    is_transposed = "MT" in parts[0]

    # Tile size in bytes: 8 rows * stride
    tile_bytes = 8 * stride

    # Within a tile, lane_id determines which 4-byte piece we get
    row_in_tile = lane_id % 8
    col_pair = lane_id // 8  # 0..3

    if is_transposed:
        # Transposed: access pattern swaps row/col addressing
        intra_tile_offset = col_pair * stride + row_in_tile * 4  # unsure, ...
        # Actually for transposed, the same addresses are used but data
        # is distributed differently.  The address each thread provides is
        # the same as non-transposed; the hardware just distributes the
        # loaded data transposed across threads.
        intra_tile_offset = row_in_tile * stride + col_pair * 4
    else:
        intra_tile_offset = row_in_tile * stride + col_pair * 4

    offsets = []
    for tile_idx in range(count):
        offsets.append(base_addr + tile_idx * tile_bytes + intra_tile_offset)

    return offsets


# ---------------------------------------------------------------------------
# Scalar decomposition helpers for HMMA on x86
# ---------------------------------------------------------------------------

def hmma_scalar_ops(shape, accum_dtype="F32"):
    """Return the scalar FMA operations needed to compute one thread's
    fragment contribution for HMMA.

    For x86 single-threaded (lane 0) emulation, HMMA is decomposed into
    scalar multiply-accumulate operations.  This function describes
    what computations lane 0 must perform.

    Parameters
    ----------
    shape : str
        "M16816", "M1688", "M884"
    accum_dtype : str
        "F32" or "F16"

    Returns
    -------
    list of dict
        Each dict describes one FMA operation:
        {
            "d_reg": int,     — destination D register index
            "a_reg": int,     — A register index
            "a_elem": int,    — element index within A register (0 or 1)
            "b_reg": int,     — B register index
            "b_elem": int,    — element index within B register (0 or 1)
            "c_reg": int,     — C register index (for accumulation)
        }

    Notes
    -----
    For m16n8k16 with f32 accumulation, lane 0:
      D[0] += A[0].lo * B[0].lo   (A[0][0] * B[0][0])
      D[0] += A[0].hi * B[0].hi   (A[0][1] * B[0][1])
      D[0] += A[2].lo * B[1].lo   (A[0][8] * B[8][0])
      D[0] += A[2].hi * B[1].hi   (A[0][9] * B[8][1])
      ... and so on for D[1], D[2], D[3]

    This gives the scalar FMA decomposition that can replace the opaque
    hmma16816f32 intrinsic with actual computation.
    """
    if shape == "M16816":
        return _hmma_m16n8k16_scalar_ops()
    elif shape == "M1688":
        return _hmma_m16n8k8_scalar_ops()
    else:
        raise ValueError(f"Scalar decomposition not supported for shape {shape!r}")


def _hmma_m16n8k16_scalar_ops():
    """Scalar FMA ops for m16n8k16, lane 0, f32 accumulation.

    D = A * B + C, where for lane 0:
      D[0] = C[0] + sum over k of A[row0][k] * B[k][col0]
      etc.

    Lane 0: groupID=0, threadID_in_group=0
      A regs: a0={A[0][0],A[0][1]}, a1={A[8][0],A[8][1]},
              a2={A[0][8],A[0][9]}, a3={A[8][8],A[8][9]}
      B regs: b0={B[0][0],B[0][1]}, b1={B[8][0],B[8][1]}
      D regs: d0=D[0][0], d1=D[8][0], d2=D[0][4], d3=D[8][4]

    D[0][0] = C[0][0] + A[0][0]*B[0][0] + A[0][1]*B[0][1]
                       + A[0][8]*B[8][0] + A[0][9]*B[8][1]
    But wait — this only accounts for k=0,1,8,9 from lane 0.
    The full k-reduction requires all 32 threads' contributions.

    For SINGLE-THREADED x86 emulation, we need ALL k values, not just
    lane 0's subset.  This is why the current lifter uses passthrough
    (returns C as-is) — a proper scalar decomposition needs to iterate
    over all lanes' data, which is not available in single-threaded mode.

    For now, we return the partial products that lane 0 can compute
    from its own registers.  A full scalar GEMM decomposition would
    require restructuring the data flow.
    """
    ops = []

    # Lane 0 partial products:
    # D[0] (d_reg=0): row=0, col=0 — accumulates over k from A row 0, B col 0
    # Partial: A reg 0 elem 0 * B reg 0 elem 0 (k=0)
    #          A reg 0 elem 1 * B reg 0 elem 1 (k=1)
    #          A reg 2 elem 0 * B reg 1 elem 0 (k=8)
    #          A reg 2 elem 1 * B reg 1 elem 1 (k=9)
    for d_reg, a_row_regs, c_reg in [
        (0, (0, 2), 0),   # D[0][0]: A row 0, B col 0
        (1, (1, 3), 1),   # D[8][0]: A row 8, B col 0
        (2, (0, 2), 2),   # D[0][4]: A row 0, B col 4
        (3, (1, 3), 3),   # D[8][4]: A row 8, B col 4
    ]:
        for a_reg, b_reg in zip(a_row_regs, (0, 1)):
            for elem in (0, 1):
                ops.append({
                    "d_reg": d_reg,
                    "a_reg": a_reg,
                    "a_elem": elem,
                    "b_reg": b_reg,
                    "b_elem": elem,
                    "c_reg": c_reg,
                })

    return ops


def _hmma_m16n8k8_scalar_ops():
    """Scalar FMA ops for m16n8k8, lane 0, f32 accumulation."""
    ops = []

    # Similar to m16n8k16 but with only 2 A regs and 1 B reg
    # A: a0={A[0][0],A[0][1]}, a1={A[8][0],A[8][1]}
    # B: b0={B[0][0],B[0][1]}
    # D: d0=D[0][0], d1=D[8][0], d2=D[0][4], d3=D[8][4]
    for d_reg, a_reg, c_reg in [
        (0, 0, 0),   # D[0][0]: A row 0, B col 0
        (1, 1, 1),   # D[8][0]: A row 8, B col 0
        (2, 0, 2),   # D[0][4]: A row 0, B col 4
        (3, 1, 3),   # D[8][4]: A row 8, B col 4
    ]:
        for elem in (0, 1):
            ops.append({
                "d_reg": d_reg,
                "a_reg": a_reg,
                "a_elem": elem,
                "b_reg": 0,
                "b_elem": elem,
                "c_reg": c_reg,
            })

    return ops


# ---------------------------------------------------------------------------
# Validation / consistency checks
# ---------------------------------------------------------------------------

def validate_fragment_coverage(shape, lane_range=None):
    """Verify that all matrix elements are covered by the fragment mapping
    across the specified lanes.

    Parameters
    ----------
    shape : str
        "M16816" or "M1688"
    lane_range : range or None
        Range of lane IDs to check. Default: range(32) for full warp.

    Returns
    -------
    dict
        Per-operand coverage report:
        {
            "A": {"covered": set, "total": (rows, cols), "complete": bool},
            "B": {"covered": set, "total": (rows, cols), "complete": bool},
            "D": {"covered": set, "total": (rows, cols), "complete": bool},
        }
    """
    if lane_range is None:
        lane_range = range(32)

    if shape == "M16816":
        dims = {"A": (16, 16), "B": (16, 8), "D": (16, 8)}
        a_nregs, b_nregs, d_nregs = 4, 2, 4
    elif shape == "M1688":
        dims = {"A": (16, 8), "B": (8, 8), "D": (16, 8)}
        a_nregs, b_nregs, d_nregs = 2, 1, 4
    else:
        raise ValueError(f"Unsupported shape for coverage check: {shape!r}")

    report = {}
    for operand, (rows, cols) in dims.items():
        covered = set()
        nregs = {"A": a_nregs, "B": b_nregs, "D": d_nregs}[operand]
        for lane in lane_range:
            for reg_idx in range(nregs):
                positions = fragment_mapping("HMMA", shape, lane, reg_idx, operand)
                for pos in positions:
                    covered.add(pos)
        total_elements = rows * cols
        report[operand] = {
            "covered": covered,
            "total": (rows, cols),
            "complete": len(covered) == total_elements,
        }

    return report
