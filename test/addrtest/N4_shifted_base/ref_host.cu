#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void N4_shifted_base(const float* in, float* out, int N, int shift) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float* p = in + shift;
    if (i >= 1 && i < N - shift - 1)
        out[i] = p[i - 1] + p[i] + p[i + 1];
}

int main(int argc, char** argv) {
    constexpr int N = 1024, shift = 10;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i%7+1)*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    N4_shifted_base<<<(N+255)/256, 256>>>(d_in, d_out, N, shift);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(h_out[0]), sizeof(h_out)/sizeof(h_out[0]), f);
        fclose(f);
    }
    bool ok = true;
    for (int i = 1; i < N - shift - 1; i++) {
        float e = h_in[shift+i-1]+h_in[shift+i]+h_in[shift+i+1];
        if (fabsf(h_out[i]-e) > 1e-5f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
