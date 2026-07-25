#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__
void tma_two_desc(const float* __restrict__ a, const float* __restrict__ b,
                  float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

int main() {
    constexpr int N = 256;
    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i + 1) * 0.3f;
        h_b[i] = (float)(N - i) * 0.2f;
    }

    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice);

    tma_two_desc<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
