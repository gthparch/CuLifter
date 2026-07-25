#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_M1_select_two_float(const float* in1, const float* in2,
                                    float* out, int N, int selector) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* src = (selector == 0) ? in1 : in2;
        out[i] = src[i];
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

    int selector = 1;  // select in2
    addr_M1_select_two_float<<<(N + 255) / 256, 256>>>(d_in1, d_in2, d_out, N, selector);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_in2[i];  // selector == 1
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
