#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__
void tma_bounds(const float* __restrict__ in, float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int left  = (idx > 0)     ? idx - 1 : 0;
        int right = (idx < N - 1) ? idx + 1 : N - 1;
        out[idx] = in[left] + in[right];
    }
}

int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.5f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    tma_bounds<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int left  = (i > 0)     ? i - 1 : 0;
        int right = (i < N - 1) ? i + 1 : N - 1;
        float expected = h_in[left] + h_in[right];
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
