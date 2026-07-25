// fa_partial_block — FlashAttention forward microbench reproducing the FA3
// hdim192_128 online-softmax PARTIAL-last-block rescale path.
//
// THE BUG UNDER TEST:
//   flash_fwd_hdim192_128_bf16_paged_sm90 fails at max_err 0.0387. V=ones->O=1.0
//   exact and Q/K reads are byte-exact, so the QK-path online-softmax max/sum
//   RESCALE across the 3 key-blocks is wrong. FA3 there uses kBlockN=96 with
//   SK=256 => 3 blocks of {96, 96, 64} — i.e. a PARTIAL last block (64 of 96).
//   The masking of the 32 padding keys + the running-max rescale across the
//   partial block is the suspect.
//
// This microbench uses the SAME tiling: KTILE=96, seqlen_k=256 => blocks
// 96,96,64 (last partial). The partial block is handled by MASKING the
// out-of-range keys to -inf BEFORE the max/exp (exactly how FA3 masks the
// padding to the 96-wide tile), so the rescale path sees a partial block.
// D=16, SQ=32 (one warp) keeps it small + fast to lift/clang, while the
// online-softmax dataflow (running max m, running denom l, corr=exp(m_old-m_new),
// partial-block mask) is identical in structure to the failing kernel.
//
// Layout: Q[SQ][D], K[SK][D], V[SK][D], O[SQ][D]  (row-major, float)
#include <cuda_runtime.h>

#define D     16
#define SQ    32
#define KTILE 96      // = FA3 hdim192 kBlockN

extern "C" __global__ void fa_partial_block(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen_k)            // = 256 (not a multiple of KTILE=96)
{
    const float scale = 0.25f;          // 1/sqrt(D=16)
    int lane = threadIdx.x;             // 0..31, query row index
    if (lane >= SQ) return;

    // Shared K/V tile staging (KTILE rows). For the partial last block the
    // tail rows hold stale data; the mask below makes them harmless.
    __shared__ float sK[KTILE][D];
    __shared__ float sV[KTILE][D];

    float q[D];
#pragma unroll
    for (int k = 0; k < D; k++) q[k] = Q[lane * D + k];

    float m = -1e30f, l = 0.0f;
    float acc[D];
#pragma unroll
    for (int k = 0; k < D; k++) acc[k] = 0.0f;

    for (int t0 = 0; t0 < seqlen_k; t0 += KTILE) {
        int tile_n = seqlen_k - t0;
        if (tile_n > KTILE) tile_n = KTILE;   // = 96,96,64 (last partial)

        // Cooperative stage: lanes load the (up to) KTILE key/value rows of
        // this block into smem. Only valid (< tile_n) rows are loaded.
        for (int r = lane; r < KTILE; r += SQ) {
            int kg = t0 + r;
            for (int k = 0; k < D; k++) {
                if (r < tile_n) {
                    sK[r][k] = K[kg * D + k];
                    sV[r][k] = V[kg * D + k];
                } else {
                    sK[r][k] = 0.0f;          // padding rows (harmless; masked)
                    sV[r][k] = 0.0f;
                }
            }
        }
        __syncthreads();

        // Scores over the FULL KTILE width; the partial-block MASK sets the
        // padding keys (j >= tile_n) to -inf before the max/exp. This is the
        // FA3 partial-block masking path under test.
        float s[KTILE];
        float tile_max = -1e30f;
        for (int j = 0; j < KTILE; j++) {
            if (j < tile_n) {
                float dot = 0.0f;
                for (int k = 0; k < D; k++) dot += q[k] * sK[j][k];
                s[j] = dot * scale;
            } else {
                s[j] = -1e30f;                // masked padding key
            }
            tile_max = fmaxf(tile_max, s[j]);
        }

        // Online rescale to the new running max (corr = exp(m_old - m_new)).
        float m_new = fmaxf(m, tile_max);
        float corr = __expf(m - m_new);
        l *= corr;
        for (int k = 0; k < D; k++) acc[k] *= corr;

        // Accumulate this block's probabilities * V (padding keys -> p≈0).
        for (int j = 0; j < KTILE; j++) {
            float p = __expf(s[j] - m_new);   // masked -> exp(-inf)=0
            l += p;
            for (int k = 0; k < D; k++) acc[k] += p * sV[j][k];
        }
        m = m_new;
        __syncthreads();
    }

    float inv = 1.0f / l;
    for (int k = 0; k < D; k++) O[lane * D + k] = acc[k] * inv;
}
