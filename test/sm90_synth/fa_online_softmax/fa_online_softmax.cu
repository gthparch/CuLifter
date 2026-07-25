// fa_online_softmax — single-warp FlashAttention forward microbenchmark.
//
// The genuine FA online-softmax algorithm (running max m, running denom l,
// rescale-on-the-fly across KV tiles) at minimal scale: ONE warp (32 threads),
// lane i owns query row i. K/V tiles are staged into shared memory and consumed
// tile-by-tile so the rescaling path (corr = exp(m_old - m_new)) is exercised.
//
// Uses only already-verified ops — FFMA (dot products), MUFU.EX2 (__expf),
// FMNMX (fmaxf), LDS/STS + BAR.SYNC (shared staging) — and NO warp-specialized
// TMA/WGMMA/mbarrier, so it lifts and runs unconditionally on the x86 backend
// (no env toggles). It isolates the FA *dataflow* (online softmax) from the
// Hopper warp-spec machinery, which is a separate concern.
//
// Layout: SQ=32 query rows, D=16 head dim, seqlen_k a multiple of 32.
//   Q[SQ][D], K[seqlen_k][D], V[seqlen_k][D], O[SQ][D]  (row-major, float)
#include <cuda_runtime.h>

#define D     16
#define SQ    32
#define KTILE 32

extern "C" __global__ void fa_online_softmax(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen_k)
{
    const float scale = 0.25f;          // 1/sqrt(D=16)
    int lane = threadIdx.x;             // 0..31, query row index
    if (lane >= SQ) return;

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
        // Cooperative stage: lane `lane` loads key/value row (t0+lane).
        for (int k = 0; k < D; k++) {
            sK[lane][k] = K[(t0 + lane) * D + k];
            sV[lane][k] = V[(t0 + lane) * D + k];
        }
        __syncthreads();

        // Scores for this tile + its local max.
        float s[KTILE];
        float tile_max = -1e30f;
        for (int j = 0; j < KTILE; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++) dot += q[k] * sK[j][k];
            s[j] = dot * scale;
            tile_max = fmaxf(tile_max, s[j]);
        }

        // Online rescale to the new running max.
        float m_new = fmaxf(m, tile_max);
        float corr = __expf(m - m_new);
        l *= corr;
        for (int k = 0; k < D; k++) acc[k] *= corr;

        // Accumulate this tile's probabilities * V.
        for (int j = 0; j < KTILE; j++) {
            float p = __expf(s[j] - m_new);
            l += p;
            for (int k = 0; k < D; k++) acc[k] += p * sV[j][k];
        }
        m = m_new;
        __syncthreads();
    }

    float inv = 1.0f / l;
    for (int k = 0; k < D; k++) O[lane * D + k] = acc[k] * inv;
}
