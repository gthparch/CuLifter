// Same as tensor_4blk4warp but with M, N, K as runtime kernel args. Mirrors
// tensor1 exactly. If this fails, the bug is in runtime-arg-derived address
// computation in the SLifter NVVM lifting.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_4blk4warp_rt(half *a, half *b, float *c, int M, int N, int K) {
    int warpInBlock = threadIdx.x / 32;
    int globalWarp = blockIdx.x * 4 + warpInBlock;
    int warpRow = globalWarp / 4;
    int warpCol = globalWarp % 4;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    wmma::load_matrix_sync(af, a + warpRow * 16 * K, K);
    wmma::load_matrix_sync(bf, b + warpCol * 16, N);
    wmma::mma_sync(cf, af, bf, cf);
    wmma::store_matrix_sync(c + warpRow * 16 * N + warpCol * 16, cf, N, wmma::mem_row_major);
}
