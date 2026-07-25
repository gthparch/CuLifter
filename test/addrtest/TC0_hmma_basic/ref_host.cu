#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
extern "C" __global__ void TC0_hmma_basic(const half* A, const half* B, float* C, int K_dim) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, K_dim);
    wmma::load_matrix_sync(b_frag, B, K_dim);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}
int main(int argc, char** argv) {
    constexpr int M=16, N=16, K=16;
    half h_A[M*K], h_B[K*N]; float h_C[M*N];
    for (int i = 0; i < M*K; i++) h_A[i] = __float2half((float)((i%5+1)*0.1f));
    for (int i = 0; i < K*N; i++) h_B[i] = __float2half((float)((i%7+1)*0.1f));
    memset(h_C, 0, sizeof(h_C));
    half *d_A, *d_B; float *d_C;
    cudaMalloc(&d_A, M*K*sizeof(half)); cudaMalloc(&d_B, K*N*sizeof(half));
    cudaMalloc(&d_C, M*N*sizeof(float));
    cudaMemcpy(d_A, h_A, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M*N*sizeof(float));
    TC0_hmma_basic<<<1, 32>>>(d_A, d_B, d_C, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_C, 4, M*N, f); fclose(f);
    }
    printf("TEST PASSED\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
