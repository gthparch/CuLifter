#include <cstdio>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
extern "C" __global__ void WS0_warp_sum(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    if (threadIdx.x % 32 == 0 && tid < N)
        out[tid / 32] = val;
}

int main(int argc, char** argv) {
    constexpr int N = 256, BS = 256;
    float h_in[N], h_out[8];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*4); cudaMalloc(&d_out, 8*4);
    cudaMemcpy(d_in, h_in, N*4, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, 8*4);
    WS0_warp_sum<<<1, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, 8*4, cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, 4, 8, f); fclose(f);
    }
    bool ok = true;
    for (int w = 0; w < 8; w++) {
        float exp = 0;
        for (int i = w*32; i < (w+1)*32 && i < N; i++) exp += h_in[i];
        if (fabsf(h_out[w] - exp) > 1.0f) { printf("FAIL warp %d: %f vs %f\n", w, h_out[w], exp); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out); return ok ? 0 : 1;
}
