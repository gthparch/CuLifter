#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void MU_exp2_raw(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out[i] = exp2f(in[i]); }
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)((i % 21) - 10);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    MU_exp2_raw<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float exp_val = exp2f(x);
        if (isnan(h_out[i]) || isinf(h_out[i]) || fabsf(h_out[i] - exp_val) > fabsf(exp_val)*1e-3f+1e-6f) {
            printf("FAIL at %d: got %f exp %f\n", i, h_out[i], exp_val); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
