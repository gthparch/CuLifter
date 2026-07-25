#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void MU_exp_chain(const float* a, const float* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = expf(a[i]) * expf(b[i]);
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) { h_a[i]=(float)((i%7)-3)*0.3f; h_b[i]=(float)((i%5)-2)*0.4f; }
    memset(h_out, 0, sizeof(h_out));
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*4); cudaMalloc(&d_b, N*4); cudaMalloc(&d_out, N*4);
    cudaMemcpy(d_a, h_a, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*4);
    MU_exp_chain<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp_val=expf(h_a[i])*expf(h_b[i]);
        if(fabsf(h_out[i]-exp_val)>fabsf(exp_val)*1e-3f+1e-6f){ printf("FAIL %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
