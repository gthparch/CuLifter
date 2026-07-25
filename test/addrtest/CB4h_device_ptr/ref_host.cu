#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

__device__ float* d_ptr;

extern "C" __global__ void CB4h_device_ptr(float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_ptr[i];
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_src[N];
    for (int i = 0; i < N; i++) h_src[i] = (float)(i*7+3);
    float *d_src;
    cudaMalloc(&d_src, N*sizeof(float));
    cudaMemcpy(d_src, h_src, N*sizeof(float), cudaMemcpyHostToDevice);
    // Set __device__ pointer to point at d_src
    cudaMemcpyToSymbol(d_ptr, &d_src, sizeof(float*));
    float *d_out; cudaMalloc(&d_out, N*sizeof(float));
    cudaMemset(d_out, 0, N*sizeof(float));
    CB4h_device_ptr<<<(N+255)/256, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    float h_out[N];
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_src[i]) { printf("FAIL at %d\n", i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_src); cudaFree(d_out); return ok ? 0 : 1;
}
