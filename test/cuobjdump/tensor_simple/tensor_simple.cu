// Simplified single-WMMA tensor test for SLifter row.row debugging.
// One 16x16x16 wmma::mma_sync call, no K-loop, single warp.
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_simple(half *a, half *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, 16);
    wmma::load_matrix_sync(bf, b, 16);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
}
