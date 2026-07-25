#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_X4_base_large_off_float(const float* in1, const float* in2,
                                    float* out, int N, int selector,
                                    unsigned long long base_offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* base = (selector == 0) ? in1 : in2;
        unsigned long long addr = (unsigned long long)base + base_offset +
                                  (unsigned long long)i * sizeof(float);
        out[i] = *(const float*)addr;
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float *h_in1 = new float[N];
    float *h_in2 = new float[N];
    float *h_out = new float[N];
    memset(h_out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_in1[i] = (float)(i % 7 + 1) * 0.1f;
        h_in2[i] = (float)((i % 5 + 1) * 0.3f);
    }

    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, N * sizeof(float));
    cudaMalloc(&d_in2, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in1, h_in1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    int selector = 0;
    unsigned long long base_offset = 0ULL;
    addr_X4_base_large_off_float<<<(N + 255) / 256, 256>>>(d_in1, d_in2, d_out, N, selector, base_offset);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_in1[i];  // selector == 0
        if (h_out[i] != expected) {
            printf("FAIL at %d: got %f expected %f\n",
                   i, h_out[i], expected);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
    delete[] h_in1; delete[] h_in2; delete[] h_out;
    return ok ? 0 : 1;
}
