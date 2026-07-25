#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void X2m_radius4(const float* const* ptrs, float* out,
                                   int N, int which) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= 4 && i < (int)(N - 4)) {
        const float* src = ptrs[which];
        float v0 = src[i + (int)(-4)];
        float v1 = src[i + (int)(-3)];
        float v2 = src[i + (int)(-2)];
        float v3 = src[i + (int)(-1)];
        float v4 = src[i];
        float v5 = src[i + (int)(1)];
        float v6 = src[i + (int)(2)];
        float v7 = src[i + (int)(3)];
        float v8 = src[i + (int)(4)];
        out[i] = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, NPTRS = 2;
    float *h_bufs[NPTRS], *d_bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        h_bufs[p] = new float[N];
        for (int i = 0; i < N; i++) h_bufs[p][i] = (float)((i % 7 + 1 + p) * 0.1f);
        cudaMalloc(&d_bufs[p], N * sizeof(float));
        cudaMemcpy(d_bufs[p], h_bufs[p], N * sizeof(float), cudaMemcpyHostToDevice);
    }
    const float* h_ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) h_ptrs[p] = d_bufs[p];
    const float** d_ptrs;
    cudaMalloc(&d_ptrs, NPTRS * sizeof(const float*));
    cudaMemcpy(d_ptrs, h_ptrs, NPTRS * sizeof(const float*), cudaMemcpyHostToDevice);
    float h_out[N], *d_out;
    cudaMalloc(&d_out, N * sizeof(float)); cudaMemset(d_out, 0, N * sizeof(float));
    int which = 0;
    X2m_radius4<<<(N+255)/256, 256>>>(d_ptrs, d_out, N, which);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 4; i < N - 4; i++) {
        float expected = h_bufs[which][i+(-4)] + h_bufs[which][i+(-3)] + h_bufs[which][i+(-2)] + h_bufs[which][i+(-1)] + h_bufs[which][i+(0)] + h_bufs[which][i+(1)] + h_bufs[which][i+(2)] + h_bufs[which][i+(3)] + h_bufs[which][i+(4)];
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    for (int p = 0; p < NPTRS; p++) { cudaFree(d_bufs[p]); delete[] h_bufs[p]; }
    cudaFree(d_ptrs); cudaFree(d_out);
    return ok ? 0 : 1;
}
