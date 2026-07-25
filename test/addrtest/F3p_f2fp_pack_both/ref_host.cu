#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3p_f2fp_pack_both(const float* a, const float* b, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __floats2half2_rn(a[i], b[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    float h_a[N], h_b[N]; half2 h_out[N];
    for (int i = 0; i < N; i++) { h_a[i] = (i%7+1)*0.3f; h_b[i] = (i%5+1)*0.7f; }
    memset(h_out, 0, sizeof(h_out));
    float *d_a, *d_b; half2 *d_out;
    cudaMalloc(&d_a, N*sizeof(float)); cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3p_f2fp_pack_both<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float lo = __half2float(__low2half(h_out[i]));
        float hi = __half2float(__high2half(h_out[i]));
        float exp_lo = __half2float(__float2half(h_a[i]));
        float exp_hi = __half2float(__float2half(h_b[i]));
        if (lo != exp_lo || hi != exp_hi) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
