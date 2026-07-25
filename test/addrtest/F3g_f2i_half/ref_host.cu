#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3g_f2i_half(const half* in, int* out_signed,
    unsigned int* out_unsigned, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_signed[i] = __half2int_rn(in[i]);
        out_unsigned[i] = __half2uint_rn(in[i]);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    half h_in[N]; int h_out_s[N]; unsigned int h_out_u[N];
    for (int i = 0; i < N; i++) h_in[i] = __float2half((float)(i%21));
    memset(h_out_s, 0, sizeof(h_out_s)); memset(h_out_u, 0, sizeof(h_out_u));
    half *d_in; int *d_os; unsigned int *d_ou;
    cudaMalloc(&d_in, N*sizeof(half)); cudaMalloc(&d_os, N*sizeof(int));
    cudaMalloc(&d_ou, N*sizeof(unsigned));
    cudaMemcpy(d_in, h_in, N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_os, 0, N*sizeof(int)); cudaMemset(d_ou, 0, N*sizeof(unsigned));
    F3g_f2i_half<<<(N+255)/256, 256>>>(d_in, d_os, d_ou, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out_s, d_os, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out_u, d_ou, N*sizeof(unsigned), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out_s, sizeof(int), N, f);
        fwrite(h_out_u, sizeof(unsigned), N, f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float fv = __half2float(h_in[i]);
        int exp_s = (int)roundf(fv);
        unsigned exp_u = (unsigned)roundf(fv);
        if (h_out_s[i] != exp_s) { printf("FAIL s at %d: got %d exp %d\n", i, h_out_s[i], exp_s); ok=false; break; }
        if (h_out_u[i] != exp_u) { printf("FAIL u at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_os); cudaFree(d_ou);
    return ok ? 0 : 1;
}
