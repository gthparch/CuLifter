#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

extern "C" __global__ void B24b_dual_pred_nested(const int* flags, const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int flag = flags[i];
        if (flag) {
            if (i < N) out[i] = in[i] * 2.0f;
        } else {
            out[i] = in[i];
        }
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024, BS = 256;
    int h_flags[N];
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_flags[i] = (i%3==0)?1:0; h_in[i] = (float)(i+1); }
    memset(h_out, 0, sizeof(h_out));
    int *d_flags; float *d_in, *d_out;
    cudaMalloc(&d_flags, N*sizeof(int));
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_flags, h_flags, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    B24b_dual_pred_nested<<<(N+BS-1)/BS, BS>>>(d_flags, d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_flags[i] ? h_in[i]*2.0f : h_in[i];
        if (fabsf(h_out[i]-exp) > 1e-4f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_flags); cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
