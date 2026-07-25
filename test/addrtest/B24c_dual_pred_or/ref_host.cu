#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B24c_dual_pred_or(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int x = i - 128;
    if (x < 0 || x >= N) return;
    out[x] = in[x] + 1.0f;
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    int total = N + 256;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*2+3);
    memset(h_out, 0, sizeof(h_out));
    float *d_in, *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    B24c_dual_pred_or<<<(total+BS-1)/BS, BS>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i] + 1.0f;
        if (fabsf(h_out[i]-exp) > 1e-5f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
