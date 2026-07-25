#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void M2m_idx_sizet(const float* const* ptrs, float* out,
                                   int N, int which) {
    size_t i = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (i < (size_t)N) {
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, NPTRS = 3;
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
    int which = 1;
    M2m_idx_sizet<<<(N+255)/256, 256>>>(d_ptrs, d_out, N, which);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_bufs[which][i]) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    for (int p = 0; p < NPTRS; p++) { cudaFree(d_bufs[p]); delete[] h_bufs[p]; }
    cudaFree(d_ptrs); cudaFree(d_out);
    return ok ? 0 : 1;
}
