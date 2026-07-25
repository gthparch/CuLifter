#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3f_i2f_half(const int* in, half* out_signed,
    const unsigned int* in_u, half* out_unsigned, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_signed[i] = __int2half_rn(in[i]);
        out_unsigned[i] = __uint2half_rn(in_u[i]);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    int h_in_s[N]; unsigned int h_in_u[N];
    half h_out_s[N], h_out_u[N];
    for (int i = 0; i < N; i++) { h_in_s[i] = (i%21)-10; h_in_u[i] = (unsigned)(i%21); }
    memset(h_out_s, 0, sizeof(h_out_s)); memset(h_out_u, 0, sizeof(h_out_u));
    int *d_s; unsigned int *d_u; half *d_os, *d_ou;
    cudaMalloc(&d_s, N*sizeof(int)); cudaMalloc(&d_u, N*sizeof(unsigned));
    cudaMalloc(&d_os, N*sizeof(half)); cudaMalloc(&d_ou, N*sizeof(half));
    cudaMemcpy(d_s, h_in_s, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_in_u, N*sizeof(unsigned), cudaMemcpyHostToDevice);
    cudaMemset(d_os, 0, N*sizeof(half)); cudaMemset(d_ou, 0, N*sizeof(half));
    F3f_i2f_half<<<(N+255)/256, 256>>>(d_s, d_os, d_u, d_ou, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_s, d_os, N*sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_u, d_ou, N*sizeof(half), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out_s, sizeof(half), N, f);
        fwrite(h_out_u, sizeof(half), N, f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float gs = __half2float(h_out_s[i]);
        if (fabsf(gs - (float)h_in_s[i]) > 0.5f) { printf("FAIL s at %d\n", i); ok=false; break; }
        float gu = __half2float(h_out_u[i]);
        if (fabsf(gu - (float)h_in_u[i]) > 0.5f) { printf("FAIL u at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_s); cudaFree(d_u); cudaFree(d_os); cudaFree(d_ou);
    return ok ? 0 : 1;
}
