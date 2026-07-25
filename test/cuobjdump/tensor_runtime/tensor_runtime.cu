// Single-warp single-mma but with M, N, K passed as runtime args (mirrors tensor1
// minus multi-warp). Used to determine if runtime args (vs hardcoded constants)
// trigger the lifter bug.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_runtime(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, K);
    wmma::load_matrix_sync(bf, b, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, N, wmma::mem_row_major);
}
