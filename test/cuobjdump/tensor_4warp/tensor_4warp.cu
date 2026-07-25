// 4-warp (single-block) single-mma_sync test: 32x32 output, 4 warps, single block.
// 4 warps in 2x2 grid; mirrors tensor1 layout but with smaller dims and a single block.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_4warp(half *a, half *b, float *c) {
    int warpId = (threadIdx.x) / 32;
    int warpRow = warpId / 2;  // 0 or 1
    int warpCol = warpId % 2;  // 0 or 1
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    // A: 32x16 row-major, stride 16. Each warp's row tile is at row warpRow*16.
    wmma::load_matrix_sync(af, a + warpRow * 16 * 16, 16);
    // B: 16x32 row-major, stride 32. Each warp's col tile is at col warpCol*16.
    wmma::load_matrix_sync(bf, b + warpCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    // C: 32x32 row-major, stride 32.
    wmma::store_matrix_sync(c + warpRow * 16 * 32 + warpCol * 16, cf, 32, wmma::mem_row_major);
}
