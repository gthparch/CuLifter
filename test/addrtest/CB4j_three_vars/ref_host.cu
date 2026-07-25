#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
__device__ int d_count = 100;
__device__ float d_scale = 0.5f;
__device__ double d_bias = 3.14159;
extern "C" __global__ void CB4j_three_vars(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N && i < d_count) out[i] = in[i] * d_scale + (float)d_bias;
}
int main(int argc, char** argv) {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    CB4j_three_vars<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < 100; i++) {
        float exp = h_in[i] * 0.5f + (float)3.14159;
        if (fabsf(h_out[i]-exp) > 1e-3f) { printf("FAIL %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
