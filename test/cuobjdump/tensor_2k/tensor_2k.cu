// 2-iteration K-loop wmma test (16x16 result, K=32 split into 2 wmma calls).
// Used to isolate K-loop accumulation bugs in the SLifter NVVM peephole.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_2k(half *a, half *b, float *c) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    // K = 32, two 16x16x16 wmma calls accumulating into the same cf.
    for (int k = 0; k < 32; k += 16) {
        wmma::load_matrix_sync(af, a + k, 32);
        wmma::load_matrix_sync(bf, b + k * 16, 16);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c, cf, 16, wmma::mem_row_major);
}
