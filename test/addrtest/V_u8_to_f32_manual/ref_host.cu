#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
extern "C" __global__ void V_u8_to_f32_manual(const int* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) out[i] = (float)(in[i] & 0xFF);
}
int main(int argc, char** argv) {
    constexpr int N = 1024;
    int h_in[N]; float h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = i*7+3;
    memset(h_out, 0, sizeof(h_out));
    int *d_in; float *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    V_u8_to_f32_manual<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) { FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f); }
    bool ok = true;
    for (int i = 0; i < N; i++) if (fabsf(h_out[i]-(float)(h_in[i]&0xFF)) > 0.5f) { printf("FAIL at %d\n", i); ok = false; break; }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
