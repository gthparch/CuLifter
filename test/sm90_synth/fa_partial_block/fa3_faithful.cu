// fa3_faithful — FlashAttention forward microbench that MIRRORS the EXACT
// FA3 hopper mainloop online-softmax STRUCTURE (mainloop_fwd_sm90 + softmax.h):
//   * REVERSE key-block iteration (n_block_max-1 .. 0); the FIRST processed
//     block is the PARTIAL/masked last block (Is_first=true, Check_inf=true).
//   * exp2 with scale-INSIDE: P = exp2(raw_score*scale_log2 - row_max*scale_log2),
//     where row_max is the RAW (unscaled) running max and scale_log2 = scale*log2(e).
//   * the SOFTWARE-PIPELINED scores_scale that is carried ONE iteration:
//     fwd_step computes max_get_scale/online_softmax for n_block but the
//     rescale_o(tOrO, scores_scale) it applies uses the PREVIOUS iter's scale.
//   * RescaleOBeforeGemm = (kHeadDim > 128): for hdim192 it is TRUE — tOrO is
//     rescaled BEFORE the PV gemm (and tOrO is clear()'d before the loop).
//   * Check_inf masking only on the first (partial) block (-INFINITY columns).
//
// This reproduces the hdim192_128 failing-kernel reduction in isolation. To
// expose the structural fault, the macros below let us toggle the FA3
// peculiarities (reverse order, RescaleOBeforeGemm, exp2-with-raw-max,
// pipelined scale) one at a time to BISECT which one diverges from the plain
// online-softmax CPU reference.
//
// D=16, SQ=32 (one warp). KTILE=96, SK=256 => 3 blocks {96,96,64} last partial.
#include <cuda_runtime.h>

#define D     16
#define SQ    32
#define KTILE 96

// --- FA3 structural toggles (default = full FA3 faithful) -------------------
#ifndef FA3_REVERSE
#define FA3_REVERSE 1          // reverse block iteration (partial block first)
#endif
#ifndef FA3_RESCALE_BEFORE
#define FA3_RESCALE_BEFORE 1   // RescaleOBeforeGemm (kHeadDim>128 -> true)
#endif
#ifndef FA3_PIPELINE_SCALE
#define FA3_PIPELINE_SCALE 1   // scores_scale carried one iter (the SW pipeline)
#endif

extern "C" __global__ void fa3_faithful(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen_k)
{
    const float scale = 0.25f;                 // softmax_scale (1/sqrt(D))
    const float scale_log2 = scale * 1.4426950408889634f;  // softmax_scale_log2

    int lane = threadIdx.x;
    if (lane >= SQ) return;

    __shared__ float sK[KTILE][D];
    __shared__ float sV[KTILE][D];

    float q[D];
#pragma unroll
    for (int k = 0; k < D; k++) q[k] = Q[lane * D + k];

    int n_block_max = (seqlen_k + KTILE - 1) / KTILE;   // ceil_div = 3

    float row_max = -1e30f;     // RAW running max (NOT scaled)
    float row_sum = 0.0f;       // running denom (l)
    float acc[D];
#pragma unroll
    for (int k = 0; k < D; k++) acc[k] = 0.0f;

    // scores_scale carried across iters (software pipeline). Init 1.0 (Is_first).
    float scores_scale = 1.0f;

    // Iterate blocks in FA3 order.
    for (int bi = 0; bi < n_block_max; bi++) {
#if FA3_REVERSE
        int n_block = n_block_max - 1 - bi;     // 2,1,0 : partial first
#else
        int n_block = bi;                       // 0,1,2 : partial last
#endif
        int t0 = n_block * KTILE;
        int tile_n = seqlen_k - t0;
        if (tile_n > KTILE) tile_n = KTILE;     // 64 for the partial block
        bool is_first = (bi == 0);
        bool check_inf = is_first;              // Seqlenk_mask only on first

        // Stage K/V for this block.
        for (int r = lane; r < KTILE; r += SQ) {
            int kg = t0 + r;
            for (int k = 0; k < D; k++) {
                if (r < tile_n) { sK[r][k] = K[kg*D+k]; sV[r][k] = V[kg*D+k]; }
                else            { sK[r][k] = 0.0f;      sV[r][k] = 0.0f; }
            }
        }
        __syncthreads();

        // QK GEMM -> RAW scores (no scale yet).
        float s[KTILE];
        for (int j = 0; j < KTILE; j++) {
            if (check_inf && j >= tile_n) {
                s[j] = -INFINITY;               // masked padding key
            } else {
                float dot = 0.0f;
                for (int k = 0; k < D; k++) dot += q[k] * sK[j][k];
                s[j] = dot;                     // RAW score
            }
        }

#if FA3_RESCALE_BEFORE && FA3_PIPELINE_SCALE
        // RescaleOBeforeGemm: rescale tOrO with the PREVIOUS scores_scale,
        // BEFORE this block's PV accumulation. (clear(tOrO) already = acc init 0.)
        for (int k = 0; k < D; k++) acc[k] *= scores_scale;
#endif

        // max_get_scale<Is_first>: update running max, compute scores_scale.
        float row_max_prev = row_max;
        for (int j = 0; j < KTILE; j++) row_max = fmaxf(row_max, s[j]);
        if (is_first) {
            scores_scale = 1.0f;
        } else {
            float rm = (row_max == -1e30f) ? 0.0f : row_max;
            scores_scale = exp2f((row_max_prev - rm) * scale_log2);
            row_sum *= scores_scale;            // rescale running denom
        }

        // online_softmax<Is_first>: P = exp2(s*scale_log2 - row_max*scale_log2).
        float row_max_scaled = (check_inf && row_max == -1e30f) ? 0.0f
                                                                 : row_max * scale_log2;
        float p[KTILE];
        float tile_sum = 0.0f;
        for (int j = 0; j < KTILE; j++) {
            p[j] = exp2f(s[j] * scale_log2 - row_max_scaled);  // -inf -> 0
            tile_sum += p[j];
        }
        if (is_first) row_sum = tile_sum;
        else          row_sum += tile_sum;

#if !(FA3_RESCALE_BEFORE && FA3_PIPELINE_SCALE)
        // Non-RescaleOBeforeGemm path: rescale AFTER computing the scale, this iter.
        if (!is_first) { for (int k = 0; k < D; k++) acc[k] *= scores_scale; }
#endif

        // PV GEMM: accumulate p*V for this block.
        for (int j = 0; j < KTILE; j++)
            for (int k = 0; k < D; k++) acc[k] += p[j] * sV[j][k];

        __syncthreads();
    }

    // finalize: O = acc / row_sum.
    float inv = (row_sum == 0.0f || row_sum != row_sum) ? 0.0f : 1.0f / row_sum;
    for (int k = 0; k < D; k++) O[lane * D + k] = acc[k] * inv;
}
