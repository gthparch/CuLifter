#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void M2m_dbl_indirect(const float* const* const* ptr_tables,
                                   float* out, int N, int table, int which) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* const* ptrs = ptr_tables[table];
        const float* src = ptrs[which];
        out[i] = src[i];
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, NTABLES = 2, NPTRS = 3;
    float *h_bufs[NTABLES][NPTRS], *d_bufs[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++) {
            h_bufs[t][p] = new float[N];
            for (int i = 0; i < N; i++) h_bufs[t][p][i] = (float)((i % 7 + 1 + t + p) * 0.1f);
            cudaMalloc(&d_bufs[t][p], N * sizeof(float));
            cudaMemcpy(d_bufs[t][p], h_bufs[t][p], N * sizeof(float), cudaMemcpyHostToDevice);
        }
    const float* h_ptrs[NTABLES][NPTRS];
    for (int t = 0; t < NTABLES; t++)
        for (int p = 0; p < NPTRS; p++) h_ptrs[t][p] = d_bufs[t][p];
    const float** d_ptrs[NTABLES];
    for (int t = 0; t < NTABLES; t++) {
        cudaMalloc(&d_ptrs[t], NPTRS * sizeof(const float*));
        cudaMemcpy(d_ptrs[t], h_ptrs[t], NPTRS * sizeof(const float*), cudaMemcpyHostToDevice);
    }
    const float** h_tables[NTABLES];
    for (int t = 0; t < NTABLES; t++) h_tables[t] = d_ptrs[t];
    const float*** d_tables;
    cudaMalloc(&d_tables, NTABLES * sizeof(const float**));
    cudaMemcpy(d_tables, h_tables, NTABLES * sizeof(const float**), cudaMemcpyHostToDevice);
    float h_out[N], *d_out;
    cudaMalloc(&d_out, N * sizeof(float)); cudaMemset(d_out, 0, N * sizeof(float));
    int table = 0, which = 1;
    M2m_dbl_indirect<<<(N+255)/256, 256>>>(d_tables, d_out, N, table, which);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_bufs[table][which][i]) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    for (int t = 0; t < NTABLES; t++) {
        for (int p = 0; p < NPTRS; p++) { cudaFree(d_bufs[t][p]); delete[] h_bufs[t][p]; }
        cudaFree(d_ptrs[t]);
    }
    cudaFree(d_tables); cudaFree(d_out);
    return ok ? 0 : 1;
}
