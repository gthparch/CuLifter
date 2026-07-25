// 4 warps, single block, runtime M/N/K. Same as tensor_4warp but with
// runtime kernel args. Isolates the multi-warp + runtime-args interaction.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_4warp_rt(half *a, half *b, float *c, int M, int N, int K) {
    int warpId = threadIdx.x / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + warpRow * 16 * K, K);
    wmma::load_matrix_sync(bf, b + warpCol * 16, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + warpRow * 16 * N + warpCol * 16, cf, N, wmma::mem_row_major);
}
