#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3j_hfma2_imm_neg(const half2* a, const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(a[i], __float2half2_rn(-1.0f), c[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    half2 h_a[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) {
        float v = (i%7+1)*0.5f;
        h_a[i] = __halves2half2(__float2half(v), __float2half(v+0.25f));
        h_c[i] = __halves2half2(__float2half(1.0f), __float2half(0.5f));
    }
    memset(h_out, 0, sizeof(h_out));
    half2 *d_a, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(half2)); cudaMalloc(&d_c, N*sizeof(half2));
    cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_a, h_a, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3j_hfma2_imm_neg<<<(N+255)/256, 256>>>(d_a, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    float imm = -1.0f;
    for (int i = 0; i < N; i++) {
        float a_lo = __half2float(__low2half(h_a[i]));
        float c_lo = __half2float(__low2half(h_c[i]));
        float o_lo = __half2float(__low2half(h_out[i]));
        if (fabsf(o_lo - (a_lo*imm + c_lo)) > 0.15f) {
            printf("FAIL at %d lo: got %f exp %f\n", i, o_lo, a_lo*imm+c_lo); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
