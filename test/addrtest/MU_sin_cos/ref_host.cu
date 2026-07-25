#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void MU_sin_cos(const float* in, float* out_sin, float* out_cos, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) { out_sin[i] = sinf(in[i]); out_cos[i] = cosf(in[i]); }
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_s[N], h_c[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 0.01f;
    memset(h_s, 0, sizeof(h_s)); memset(h_c, 0, sizeof(h_c));
    float *d_in, *d_s, *d_c;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_s, N*4); cudaMalloc(&d_c, N*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_s, 0, N*4); cudaMemset(d_c, 0, N*4);
    MU_sin_cos<<<(N+255)/256, 256>>>(d_in, d_s, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_s, d_s, N*4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c, d_c, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_s, 4, N, f); fwrite(h_c, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_s[i]-sinf(h_in[i]))>1e-5f||fabsf(h_c[i]-cosf(h_in[i]))>1e-5f) {
            printf("FAIL at %d\n",i); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_s); cudaFree(d_c);
    return ok ? 0 : 1;
}
