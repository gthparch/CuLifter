#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void P2d_load_sub_u32(const unsigned int* data, const int* offsets,
    unsigned int* out, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int off = offsets[i];
        int idx = off - K;
        if (idx >= 0 && idx < N)
            out[i] = data[idx];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, K = 3;
    unsigned int h_data[N]; int h_off[N]; unsigned int h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_data[i] = (uint32_t)((i%7+1)*100); h_off[i] = (i*7+K) % N; }
    unsigned int *d_data, *d_out; int *d_off;
    cudaMalloc(&d_data, N*sizeof(uint32_t)); cudaMalloc(&d_off, N*sizeof(int));
    cudaMalloc(&d_out, N*sizeof(uint32_t));
    cudaMemcpy(d_data, h_data, N*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_off, h_off, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(uint32_t));
    P2d_load_sub_u32<<<(N+255)/256, 256>>>(d_data, d_off, d_out, N, K);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(uint32_t), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int idx = h_off[i] - K;
        if (idx >= 0 && idx < N)
            if (h_out[i] != h_data[idx]) { printf("FAIL at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_data); cudaFree(d_off); cudaFree(d_out);
    return ok ? 0 : 1;
}
