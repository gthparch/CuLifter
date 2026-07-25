// EXACT mirror of tensor.cu's structure: warpRow/warpCol computed from
// runtime warpM/warpN division. If this fails (where tensor_kloop_grid
// passes), the issue is the runtime M/N integer divide in address calc.
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_kloop_runtime(half *a, half *b, float *c, int M, int N, int K) {
    const int warpM = (M + 16 - 1) / 16;
    const int warpN = (N + 16 - 1) / 16;
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int warpRow = warpId / warpN;
    const int warpCol = warpId % warpN;
    if (warpRow >= warpM || warpCol >= warpN) return;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(af, a + warpRow * 16 * K + k, K);
        wmma::load_matrix_sync(bf, b + k * N + warpCol * 16, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(c + warpRow * 16 * N + warpCol * 16, cf, N, wmma::mem_row_major);
}
