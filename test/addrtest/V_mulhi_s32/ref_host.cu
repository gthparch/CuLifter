#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_mulhi_s32(const int* a, const int* b, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = (int)(((long long)a[i] * (long long)b[i]) >> 32);
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    int h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i] = (i*12345)-500000; h_b[i] = (i*54321)-300000; }
    memset(h_out, 0, sizeof(h_out));
    int *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*4); cudaMalloc(&d_b, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_a, h_a, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    V_mulhi_s32<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) { FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f); }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int exp = (int)(((long long)h_a[i] * (long long)h_b[i]) >> 32);
        if (h_out[i] != exp) { printf("FAIL at %d\n", i); ok = false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
