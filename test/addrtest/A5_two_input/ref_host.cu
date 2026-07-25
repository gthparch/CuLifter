#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void A5_two_input(const float* __restrict__ in1,
    const float* __restrict__ in2, float* __restrict__ out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = in1[i] + in2[i];
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in1[N], h_in2[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) { h_in1[i] = (float)(i%7+1)*0.1f; h_in2[i] = (float)(i%5+1)*0.2f; }
    float *d_in1, *d_in2, *d_out;
    cudaMalloc(&d_in1, N*sizeof(float)); cudaMalloc(&d_in2, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in1, h_in1, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in2, h_in2, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    A5_two_input<<<(N+255)/256, 256>>>(d_in1, d_in2, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - (h_in1[i]+h_in2[i])) > 1e-5f) { printf("FAIL at %d\n", i); ok = false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in1); cudaFree(d_in2); cudaFree(d_out);
    return ok ? 0 : 1;
}
