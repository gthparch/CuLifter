#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__
void tma_1d_offset(const float* __restrict__ in, float* __restrict__ out,
                   int N, int offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int src = idx + offset;
        if (src >= 0 && src < N)
            out[idx] = in[src];
        else
            out[idx] = 0.0f;
    }
}

int main() {
    constexpr int N = 256;
    constexpr int OFFSET = 3;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.5f;
    memset(h_out, 0, sizeof(h_out));

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    tma_1d_offset<<<(N+255)/256, 256>>>(d_in, d_out, N, OFFSET);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected;
        int src = i + OFFSET;
        if (src >= 0 && src < N) expected = h_in[src];
        else expected = 0.0f;
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n", i, h_out[i], expected);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
