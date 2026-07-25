#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_A1_copy_1d_double(const double* in, double* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    double *h_in = new double[N];
    double *h_out = new double[N];
    memset(h_out, 0, N * sizeof(double));
    for (int i = 0; i < N; i++) h_in[i] = (double)(double)(i % 7 + 1) * 0.1f;

    double *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(double));
    cudaMalloc(&d_out, N * sizeof(double));
    cudaMemcpy(d_in, h_in, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(double));

    addr_A1_copy_1d_double<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(double), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) {
            printf("FAIL at %d: got %f expected %f\n",
                   i, h_out[i], h_in[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
