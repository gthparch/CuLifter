// Multi-block test: 4 blocks × 1 warp each = 4 warps total. 32x32 result.
// Each block has 1 warp computing one 16x16 tile. blockIdx.x picks the tile.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_4block(half *a, half *b, float *c) {
    int blkId = blockIdx.x;
    int blkRow = blkId / 2;  // 0 or 1
    int blkCol = blkId % 2;  // 0 or 1
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + blkRow * 16 * 16, 16);
    wmma::load_matrix_sync(bf, b + blkCol * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + blkRow * 16 * 32 + blkCol * 16, cf, 32, wmma::mem_row_major);
}
