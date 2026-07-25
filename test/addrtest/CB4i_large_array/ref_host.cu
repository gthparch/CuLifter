#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
__device__ float d_lut[1024];
extern "C" __global__ void CB4i_large_array(float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = d_lut[i] + 1.0f;
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_lut[N]; for (int i = 0; i < N; i++) h_lut[i] = (float)(i*3+7);
    cudaMemcpyToSymbol(d_lut, h_lut, N*sizeof(float));
    float h_out[N]; memset(h_out, 0, sizeof(h_out));
    float *d_out; cudaMalloc(&d_out, N*4); cudaMemset(d_out, 0, N*4);
    CB4i_large_array<<<(N+255)/256, 256>>>(d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - (h_lut[i]+1.0f)) > 1e-4f) { printf("FAIL %d\n",i); ok=false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_out); return ok ? 0 : 1;
}
