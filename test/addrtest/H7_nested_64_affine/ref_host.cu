#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void H7_nested_64_affine(const float* in, float* out,
    unsigned long long S0, unsigned long long S1, int N) {
    unsigned long long b0 = blockIdx.x;
    unsigned long long b1 = threadIdx.x;
    unsigned long long idx = ((b0 * S0 + b1) * S1 + b1);
    int i = (int)(idx % (unsigned long long)N);
    if (i < N)
        out[i] = in[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));
    H7_nested_64_affine<<<1, 256>>>(d_in, d_out, 1048576ULL, 65536ULL, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < 256; i++) {
        unsigned long long idx = ((0ULL * 1048576ULL + (unsigned long long)i) * 65536ULL + (unsigned long long)i);
        int j = (int)(idx % 1024ULL);
        if (h_out[j] != h_in[j]) {
            printf("FAIL at %d\n", j); ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
