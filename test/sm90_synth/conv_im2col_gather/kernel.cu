// CUTLASS group-conv im2col ACTIVATION GATHER audit (42_ampere_tensorop_group_conv).
//
// Faithfully reproduces cutlass::conv::threadblock::
//   Conv2dFpropActivationTileAccessIteratorAnalytic  (GroupMode::kSingleGroup)
// from include/cutlass/conv/threadblock/
//   conv2d_fprop_activation_tile_access_iterator_analytic.h
//
// The implicit-GEMM A-matrix has rows = N*P*Q (output spatial positions, "npq")
// and cols = the contraction dimension walked as (filter_r, filter_s, filter_c)
// triples ("crs"). For each (npq_row, crs_step) the iterator computes the source
// activation coordinate and gathers Activation[NHWC offset], with PREDICATION
// (out-of-bounds -> 0) and a CHANNEL-GROUP SPLIT advance (the distinguishing
// surface vs the plain GEMM examples).
//
// This kernel computes, for each thread = one (npq_row, crs_step) cell:
//   1. npq decompose:  n = npq/(P*Q); res = npq%(P*Q); p = res/Q; q = res%Q   <- runtime div/mod
//   2. crs walk with GROUP ADVANCE: filter_c starts at filter_c_init; every
//      crs_per_group steps it jumps to group_idx*channels_per_group + filter_c_init
//   3. at(): h = p*stride_h - pad_h + r*dilation_h ;  w = q*stride_w - pad_w + s*dil_w
//   4. valid(): n<N && 0<=h<H && 0<=w<W && c<C  (predicate)
//   5. NHWC offset = c + C*w + (W*C)*h + (H*W*C)*n   (TensorNHWC operator())
//   6. gather Activation[offset] (valid) else 0  -> write to GatherOut[row*CRS + step]
//
// This isolates the gather ADDRESSING + predication + group channel split, the
// distinct surface; the HMMA/epilogue are audited elsewhere. Activation is f16,
// gather output stored as f32 (lossless) so we value-verify bit-exactly.
//
// Params (NHWC group-conv geometry) passed as scalars + 2 pointers.
#include <cuda_fp16.h>

extern "C" __global__ void kern_im2col_gather(
    const __half* __restrict__ Activation,   // [N][H][W][C] NHWC packed
    float* __restrict__ GatherOut,           // [NPQ][CRS] compacted im2col matrix (f32)
    int N, int H, int W, int C,
    int P, int Q,                            // output spatial (rows = N*P*Q)
    int R, int S,                            // filter spatial
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int groups,
    int Shape_kColumn,                       // threadblock tile N (channel tile, =64)
    int CRS)                                 // total contraction steps = R*S*ceil(cpg/kColumn)*kColumn ... (precomputed)
{
    // Flat thread id = one (npq_row, crs_step) cell of the implicit-GEMM A matrix.
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long NPQ  = (long)N * P * Q;
    const long total = NPQ * (long)CRS;
    if (tid >= total) return;

    const int npq_row  = (int)(tid / CRS);
    const int crs_step = (int)(tid % CRS);

    // --- npq decompose (iterator ctor; RUNTIME div/mod when P*Q, Q not pow2) ---
    const int PQ = P * Q;
    const int n  = npq_row / PQ;
    const int res = npq_row % PQ;
    const int p  = res / Q;
    const int q  = res % Q;

    // --- channel-group geometry (GroupMode::kSingleGroup) ---
    const int channels_per_group = C / groups;
    const int crs_per_group = S * R * ((channels_per_group + Shape_kColumn - 1) / Shape_kColumn);

    // --- walk the crs dimension to step crs_step, mirroring advance() exactly ---
    // State: filter_r, filter_s (spatial), filter_c (channel), crs_cnt, group_idx.
    // advance() order: ++crs_cnt; ++s; if s<S return; s=0; ++r; if r<R return; r=0;
    //   then channel/group bump. We REPLAY advance() crs_step times from init.
    // filter_c_init = thread_coord.contiguous() = 0 here (single-tile, col 0 lane 0
    //   conceptual: we sweep the whole channel block so init=0).
    const int filter_c_init = 0;
    int filter_r = 0, filter_s = 0, filter_c = filter_c_init;
    int crs_cnt = 0, group_idx = 0;
    for (int step = 0; step < crs_step; ++step) {
        ++crs_cnt;                 // GroupMode != kNone: ++crs_cnt every advance
        ++filter_s;
        if (filter_s < S) continue;
        filter_s = 0;
        ++filter_r;
        if (filter_r < R) continue;
        filter_r = 0;
        // channel / group advance
        if (crs_cnt == crs_per_group) {
            crs_cnt = 0;
            ++group_idx;
            filter_c = group_idx * channels_per_group + filter_c_init;
        } else {
            filter_c += Shape_kColumn;   // split_k_slices = 1
        }
    }

    // --- at(): map (p,q,filter_r,filter_s,filter_c) -> activation (n,h,w,c) ---
    // Mode::kCrossCorrelation (default for fprop): r=filter_r, s=filter_s.
    const int r = filter_r;
    const int s = filter_s;
    const int h = p * stride_h - pad_h + r * dilation_h;
    const int w = q * stride_w - pad_w + s * dilation_w;
    const int c = filter_c;   // iteration_vector*kElements folded out (kElements gather as scalar here)

    // --- valid() predicate ---
    const bool valid = (n < N) && (h >= 0) && (h < H) && (w >= 0) && (w < W) && (c < C);

    float val = 0.f;
    if (valid) {
        // TensorNHWC offset = c + C*w + (W*C)*h + (H*W*C)*n
        const long off = (long)c
                       + (long)C * w
                       + (long)(W * C) * h
                       + (long)(H * W * C) * n;
        val = __half2float(Activation[off]);
    }
    GatherOut[(long)npq_row * CRS + crs_step] = val;
}
