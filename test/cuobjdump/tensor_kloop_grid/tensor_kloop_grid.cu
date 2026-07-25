// 4 blocks × 4 warps with K-loop and runtime args: M=N=K=64.
// Adds K-loop on top of tensor_4blk4warp_rt. If this fails, the bug is
// in K-loop accumulation across multi-block/multi-warp + runtime args.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_kloop_grid(half *a, half *b, float *c, int M, int N, int K) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 4;  // N/16 = 4 col tiles per row
    int warpCol = warpId % 4;
    if (warpRow >= 4 || warpCol >= 4) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + warpRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + warpCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + warpRow * 16 * N + warpCol * 16, cf, N, wmma::mem_row_major);
}
