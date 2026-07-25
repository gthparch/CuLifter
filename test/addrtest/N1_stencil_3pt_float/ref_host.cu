#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void addr_N1_stencil_3pt_float(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1)
        out[i] = in[i - 1] + in[i] + in[i + 1];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float *h_in = new float[N];
    float *h_out = new float[N];
    memset(h_out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N * sizeof(float));

    addr_N1_stencil_3pt_float<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(float), N, f);
        fclose(f);
    }

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        float expected = h_in[i - 1] + h_in[i] + h_in[i + 1];
        if (fabsf(h_out[i] - expected) > 1e-5f) {
            printf("FAIL at %d: got %f expected %f\n",
                   i, h_out[i], expected);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");

    cudaFree(d_in); cudaFree(d_out);
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
