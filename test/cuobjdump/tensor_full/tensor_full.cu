// Mirrors tensor.cu's structure exactly: 4 blocks × 4 warps, M=N=K=256,
// 16 K-iterations per warp. Used to verify that the kernel-arg fix also
// resolves tensor.cu's failure.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_full(half *a, half *b, float *c, int M, int N, int K) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int warpRow = warpId / 16;  // M/16 = 16 col tiles per row
    int warpCol = warpId % 16;
    if (warpRow >= 16 || warpCol >= 16) return;

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
