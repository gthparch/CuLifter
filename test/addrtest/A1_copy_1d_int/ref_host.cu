#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_A1_copy_1d_int(const int* in, int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    int *h_in = new int[N];
    int *h_out = new int[N];
    memset(h_out, 0, N * sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = (int)(i % 7 + 1);

    int *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_out, N * sizeof(int));
    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(int));

    addr_A1_copy_1d_int<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) {
            printf("FAIL at %d: got %d expected %d\n",
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
