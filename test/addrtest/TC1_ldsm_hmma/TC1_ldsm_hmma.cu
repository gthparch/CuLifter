#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void TC1_ldsm_hmma(const half* A, const half* B, float* C,
    int M, int N, int K_dim) {
    __shared__ half As[16][16];
    __shared__ half Bs[16][16];
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    // Cooperative load to shared
    if (ty < 16 && tx < 16) {
        As[ty][tx] = A[ty * K_dim + tx];
        Bs[ty][tx] = B[ty * N + tx];
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, &As[0][0], 16);
    wmma::load_matrix_sync(b_frag, &Bs[0][0], 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
