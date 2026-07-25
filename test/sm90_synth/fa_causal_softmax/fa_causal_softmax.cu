// fa_causal_softmax — single-warp CAUSAL FlashAttention forward microbench.
//
// Same online-softmax core as fa_online_softmax, plus the causal mask: query
// row i only attends to key positions j <= i. With SQ=32 and seqlen_k=64 this
// exercises (a) a PARTIALLY masked tile (keys 0..31) and (b) a FULLY masked KV
// tile (keys 32..63, all > any query index) — the latter must leave the running
// max/denominator unchanged (tile_max=-inf -> corr=exp(m-m)=1, p=exp(-inf)=0).
// The mask is a predicated score override (ISETP + select to -inf), the path
// real causal FA kernels take. Verified vs a CPU causal-attention oracle.
//
// Layout: SQ=32 query rows, D=16, seqlen_k multiple of 32; Q/K/V/O row-major f32.
#include <cuda_runtime.h>

#define D     16
#define SQ    32
#define KTILE 32

extern "C" __global__ void fa_causal_softmax(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seqlen_k)
{
    const float scale = 0.25f;          // 1/sqrt(D=16)
    int lane = threadIdx.x;             // query row index
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
        for (int k = 0; k < D; k++) {
            sK[lane][k] = K[(t0 + lane) * D + k];
            sV[lane][k] = V[(t0 + lane) * D + k];
        }
        __syncthreads();

        float s[KTILE];
        float tile_max = -1e30f;
        for (int j = 0; j < KTILE; j++) {
            float dot = 0.0f;
            for (int k = 0; k < D; k++) dot += q[k] * sK[j][k];
            float sj = dot * scale;
            // Causal mask: key position (t0+j) may not exceed query position.
            sj = ((t0 + j) > lane) ? -1e30f : sj;
            s[j] = sj;
            tile_max = fmaxf(tile_max, sj);
        }

        float m_new = fmaxf(m, tile_max);
        float corr = __expf(m - m_new);
        l *= corr;
        for (int k = 0; k < D; k++) acc[k] *= corr;

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
