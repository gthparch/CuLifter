#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void P2b_sel_sub_u32(const unsigned int* A, const unsigned int* B,
    unsigned int* out, int N, int selector, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= offset && i < N) {
        const unsigned int* src = (selector == 0) ? A : B;
        out[i] = src[i - offset];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, offset = 4;
    unsigned int h_A[N], h_B[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_A[i] = (unsigned int)((i%7+1)*0.1f); h_B[i] = (unsigned int)((i%5+1)*0.2f); }
    unsigned int *d_A, *d_B, *d_out;
    cudaMalloc(&d_A, N*sizeof(uint32_t)); cudaMalloc(&d_B, N*sizeof(uint32_t)); cudaMalloc(&d_out, N*sizeof(uint32_t));
    cudaMemcpy(d_A, h_A, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(uint32_t));
    int selector = 1;
    P2b_sel_sub_u32<<<(N+255)/256, 256>>>(d_A, d_B, d_out, N, selector, offset);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint32_t), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = offset; i < N; i++)
        if (h_out[i] != h_B[i-offset]) { printf("FAIL at %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_out);
    return ok ? 0 : 1;
}
