#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_X2_ptr_stencil_float(const float* const* ptrs, float* out,
                                    int N, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1) {
        const float* src = ptrs[which];
        out[i] = src[i - 1] + src[i] + src[i + 1];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    constexpr int NPTRS = 2;
    float *h_bufs[NPTRS];
    float *d_bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        h_bufs[p] = new float[N];
        for (int i = 0; i < N; i++)
            h_bufs[p][i] = (float)((i % 7 + 1 + p) * 0.1f);
        cudaMalloc(&d_bufs[p], N * sizeof(float));
        cudaMemcpy(d_bufs[p], h_bufs[p], N * sizeof(float), cudaMemcpyHostToDevice);
    }

    const float* h_ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) h_ptrs[p] = d_bufs[p];
    const float** d_ptrs;
    cudaMalloc(&d_ptrs, NPTRS * sizeof(const float*));
    cudaMemcpy(d_ptrs, h_ptrs, NPTRS * sizeof(const float*), cudaMemcpyHostToDevice);

    float *h_out = new float[N];
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemset(d_out, 0, N * sizeof(float));

    int which = 0;
    addr_X2_ptr_stencil_float<<<(N + 255) / 256, 256>>>(d_ptrs, d_out, N, which);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        float expected = h_bufs[which][i-1] + h_bufs[which][i] + h_bufs[which][i+1];
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n",
                   i, h_out[i], expected);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    for (int p = 0; p < NPTRS; p++) { cudaFree(d_bufs[p]); delete[] h_bufs[p]; }
    cudaFree(d_ptrs); cudaFree(d_out);
    delete[] h_out;
    return ok ? 0 : 1;
}
