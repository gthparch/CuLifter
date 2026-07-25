// Tensor-core (WMMA) microbenchmark suite. 15 small kernels that bisect the
// SLifter NVVM lift across (single vs multi)-warp, (single vs multi)-block,
// (constant vs runtime) kernel args, K-loop unroll vs phi accumulator, etc.
// Used to localize and regression-test the WMMA peephole + kernel-arg fixes.
//
// Each kernel is independent; host_cuda.cu drives every one in sequence and
// reports a single TEST PASSED line if all match a CPU reference, or FAIL
// with the offending kernel name + index/value otherwise.
#include <mma.h>
using namespace nvcuda;

// ============================================================================
// Single warp, single mma_sync, hardcoded constants — baseline.
// ============================================================================
extern "C" __global__ void tm_simple(half *a, half *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
}

// ============================================================================
// Single warp, K-loop unrolled (constant trip count K=32).
// ============================================================================
extern "C" __global__ void tm_2k(half *a, half *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < 32; k += 16) {
        wmma::load_matrix_sync(af, a + k, 32);
        wmma::load_matrix_sync(bf, b + k * 16, 16);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
}

// ============================================================================
// 2 warps in a single block, hardcoded constants.
// ============================================================================
extern "C" __global__ void tm_2warp(half *a, half *b, float *c) {
    int warpId = threadIdx.x / 32;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b + warpId * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + warpId * 16, cf, 32, wmma::mem_row_major);
}

// ============================================================================
// 4 warps in a single block, hardcoded constants (32x32 result).
// ============================================================================
extern "C" __global__ void tm_4warp(half *a, half *b, float *c) {
    int warpId = threadIdx.x / 32;
    int wRow = warpId / 2, wCol = warpId % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + wRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + wCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + wRow * 16 * 32 + wCol * 16, cf, 32, wmma::mem_row_major);
}

// ============================================================================
// 4 blocks of 1 warp each, hardcoded constants.
// ============================================================================
extern "C" __global__ void tm_4block(half *a, half *b, float *c) {
    int blkRow = blockIdx.x / 2, blkCol = blockIdx.x % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + blkRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + blkCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + blkRow * 16 * 32 + blkCol * 16, cf, 32, wmma::mem_row_major);
}

// ============================================================================
// Single warp + runtime kernel args (M=N=K=16). Verifies arg slot wiring.
// ============================================================================
extern "C" __global__ void tm_runtime(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, K);
    wmma::load_matrix_sync(bf, b, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, N, wmma::mem_row_major);
}

// ============================================================================
// Single warp + runtime args with K != N stride (M=N=16, K=32 K-loop). Caught
// the "unused M dropped from IR signature" bug — without the fix, N lands in
// K's slot and A's load uses the wrong stride.
// ============================================================================
extern "C" __global__ void tm_runtime_kne(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + k, K);
        wmma::load_matrix_sync(bf, b + k * N, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c, cf, N, wmma::mem_row_major);
}

// ============================================================================
// 2 blocks × 2 warps = 4 warps, hardcoded constants.
// ============================================================================
extern "C" __global__ void tm_2blk2warp(half *a, half *b, float *c) {
    int wInBlk = threadIdx.x / 32;
    int gw = blockIdx.x * 2 + wInBlk;
    int wRow = gw / 2, wCol = gw % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + wRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + wCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + wRow * 16 * 32 + wCol * 16, cf, 32, wmma::mem_row_major);
}

// ============================================================================
// 4 blocks × 4 warps = 16 warps, hardcoded (M=N=64, K=16). Stresses grid
// scheduling.
// ============================================================================
extern "C" __global__ void tm_4blk4warp(half *a, half *b, float *c) {
    int wInBlk = threadIdx.x / 32;
    int gw = blockIdx.x * 4 + wInBlk;
    int wRow = gw / 4, wCol = gw % 4;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + wRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + wCol * 16, 64);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + wRow * 16 * 64 + wCol * 16, cf, 64, wmma::mem_row_major);
}

// ============================================================================
// 4 warps single block + runtime args (no K-loop, no warpN division).
// ============================================================================
extern "C" __global__ void tm_4warp_rt(half *a, half *b, float *c, int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int wRow = warpId / 2, wCol = warpId % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + wRow * 16 * K, K);
    wmma::load_matrix_sync(bf, b + wCol * 16, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}

// ============================================================================
// 4 blocks × 4 warps + runtime args (no K-loop, no warpN division).
// ============================================================================
extern "C" __global__ void tm_4blk4warp_rt(half *a, half *b, float *c, int M, int N, int K) {
    int wInBlk = threadIdx.x / 32;
    int gw = blockIdx.x * 4 + wInBlk;
    int wRow = gw / 4, wCol = gw % 4;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + wRow * 16 * K, K);
    wmma::load_matrix_sync(bf, b + wCol * 16, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}

// ============================================================================
// Single block 4 warps + K-loop (runtime K, hardcoded warpN).
// ============================================================================
extern "C" __global__ void tm_kloop_4warp(half *a, half *b, float *c, int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int wRow = warpId / 2, wCol = warpId % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + wRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + wCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}

// ============================================================================
// 4 blocks × 4 warps + K-loop (runtime K, hardcoded warpN). Same grid as
// tensor.cu's structure but without runtime warpN division.
// ============================================================================
extern "C" __global__ void tm_kloop_grid(half *a, half *b, float *c, int M, int N, int K) {
    int wInBlk = threadIdx.x / 32;
    int gw = blockIdx.x * 4 + wInBlk;
    int wRow = gw / 4, wCol = gw % 4;
    if (wRow >= 4 || wCol >= 4) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + wRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + wCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}

// ============================================================================
// Mirror of tensor.cu: 4 blocks × 4 warps with M=N=K=256 hardcoded constants
// (passes — proves the bug is RUNTIME divides, not the size).
// ============================================================================
extern "C" __global__ void tm_full(half *a, half *b, float *c, int M, int N, int K) {
    int wInBlk = threadIdx.x / 32;
    int gw = blockIdx.x * 4 + wInBlk;
    int wRow = gw / 16, wCol = gw % 16;
    if (wRow >= 16 || wCol >= 16) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + wRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + wCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}

// ============================================================================
// EXACT mirror of tensor.cu's structure (warpRow = gw / warpN where warpN is
// derived from runtime N via integer divide). FAILS today because the lifter
// mishandles ptxas's reciprocal-multiplication pattern for runtime divides;
// see division_microbench.cu for isolation kernels.
// ============================================================================
extern "C" __global__ void tm_kloop_runtime(half *a, half *b, float *c, int M, int N, int K) {
    const int warpM = (M + 16 - 1) / 16;
    const int warpN = (N + 16 - 1) / 16;
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int wRow = warpId / warpN;
    const int wCol = warpId % warpN;
    if (wRow >= warpM || wCol >= warpN) return;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + wRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + wCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + wRow * 16 * N + wCol * 16, cf, N, wmma::mem_row_major);
}
