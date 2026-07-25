#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__
void tma_loop(const float* __restrict__ in, float* __restrict__ out,
              int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int base = idx * K;
        for (int j = 0; j < K; j++) {
            sum += in[base + j];
        }
        out[idx] = sum;
    }
}

int main() {
    constexpr int N = 64;
    constexpr int K = 4;
    constexpr int IN_SIZE = N * K;
    float h_in[IN_SIZE], h_out[N];
    for (int i = 0; i < IN_SIZE; i++) h_in[i] = (float)(i + 1) * 0.01f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, IN_SIZE * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, IN_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    tma_loop<<<(N+255)/256, 256>>>(d_in, d_out, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = 0.0f;
        int base = i * K;
        for (int j = 0; j < K; j++) expected += h_in[base + j];
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
