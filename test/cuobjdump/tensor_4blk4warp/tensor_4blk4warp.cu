// 4 blocks × 4 warps each = 16 warps total. 64x64 result. Mirrors tensor1's
// grid layout exactly but with hardcoded constants instead of runtime args.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_4blk4warp(half *a, half *b, float *c) {
    int warpInBlock = threadIdx.x / 32;
    int globalWarp = blockIdx.x * 4 + warpInBlock;
    int warpRow = globalWarp / 4;  // 0..3 (since 4 col tiles per row)
    int warpCol = globalWarp % 4;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + warpRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + warpCol * 16, 64);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + warpRow * 16 * 64 + warpCol * 16, cf, 64, wmma::mem_row_major);
}
