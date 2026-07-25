#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

__device__ float d_bias = 100.0f;

extern "C" __global__ void CB4f_mixed_c0_c4(const float* in, float* out, int N, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = in[i] * scale + d_bias;
}

int main(int argc, char** argv) {
    constexpr int N = 256;
    float scale = 2.0f;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1) * 0.5f;
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    CB4f_mixed_c0_c4<<<(N+255)/256, 256>>>(d_in, d_out, N, scale);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i] * scale + 100.0f;
        if (fabsf(h_out[i] - exp) > 1e-3f) { printf("FAIL at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
