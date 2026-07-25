#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3e_f2f_roundtrip(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half h = __float2half(in[i]);
        out[i] = __half2float(h);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_in[N], h_out[N];
    float vals[] = {0.1f, 0.3333f, 1.5f, 65504.0f, 0.000061f, -0.5f, 100.25f, 3.14159f};
    for (int i = 0; i < N; i++) h_in[i] = vals[i%8] * ((i/8)%2==0 ? 1.0f : -1.0f);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    F3e_f2f_roundtrip<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        half h = __float2half(h_in[i]);
        float exp = __half2float(h);
        if (h_out[i] != exp) { printf("FAIL at %d\n", i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
