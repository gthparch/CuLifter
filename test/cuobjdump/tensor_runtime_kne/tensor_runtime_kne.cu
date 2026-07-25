// Single-warp single-mma_sync but with M=16, N=32, K=16 as runtime args
// (K != N). If this fails, the bug is in single-warp lifting when N != K
// (not multi-warp specific). If it passes, the bug really is multi-warp + K!=N.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_runtime_kne(half *a, half *b, float *c, int M, int N, int K) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a, K);
    wmma::load_matrix_sync(bf, b, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c, cf, N, wmma::mem_row_major);
}
