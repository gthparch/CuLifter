#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3q_f2fp_single_slot(const float* in, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half h = __float2half(in[i]);
        out[i] = __halves2half2(h, __float2half(0.0f));
    }
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    float h_in[N]; half2 h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (i%11+1)*0.3f;
    memset(h_out, 0xFF, sizeof(h_out));
    float *d_in; half2 *d_out;
    cudaMalloc(&d_in, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3q_f2fp_single_slot<<<(N+255)/256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        half exp_lo = __float2half(h_in[i]);
        float lo = __half2float(__low2half(h_out[i]));
        float hi = __half2float(__high2half(h_out[i]));
        if (lo != __half2float(exp_lo) || hi != 0.0f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
