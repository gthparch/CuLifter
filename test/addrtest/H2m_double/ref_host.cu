#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void H2m_double(const double* in, double* out, int N) {
    size_t idx = (size_t)blockIdx.x * 1048576ULL + (size_t)threadIdx.x;
    int i = (int)(idx % (size_t)N);
    if (i < N)
        out[i] = in[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    double h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (double)((i % 7 + 1) * 0.1f);
    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double)); cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(double));
    H2m_double<<<1, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < 256; i++) {
        if (h_out[i] != h_in[i]) {
            printf("FAIL at %d\n", i); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
