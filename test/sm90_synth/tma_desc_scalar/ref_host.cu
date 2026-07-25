#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__
void tma_desc_scalar(const float* __restrict__ in, float* __restrict__ out,
                     int N, float alpha, float beta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = in[idx] * alpha + beta;
    }
}

int main() {
    constexpr int N = 256;
    float alpha = 2.5f, beta = 1.0f;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    tma_desc_scalar<<<(N+255)/256, 256>>>(d_in, d_out, N, alpha, beta);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_in[i] * alpha + beta;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
