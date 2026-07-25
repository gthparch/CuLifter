// 2 blocks × 2 warps each = 4 warps total. 32x32 result, single mma per warp.
// Mirrors tensor1 grid structure (multi-block × multi-warp) at smaller scale.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_2blk2warp(half *a, half *b, float *c) {
    int warpInBlock = threadIdx.x / 32;
    int globalWarp = blockIdx.x * 2 + warpInBlock;
    int warpRow = globalWarp / 2;
    int warpCol = globalWarp % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + warpRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + warpCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + warpRow * 16 * 32 + warpCol * 16, cf, 32, wmma::mem_row_major);
}
