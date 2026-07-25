#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3m_hfma2_two_imm(const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(__float2half2_rn(0.5f), __float2half2_rn(0.5f), c[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    half2 h_c[N], h_out[N];
    for (int i = 0; i < N; i++)
        h_c[i] = __halves2half2(__float2half((i%7+1)*0.1f), __float2half((i%5+1)*0.2f));
    memset(h_out, 0, sizeof(h_out));
    half2 *d_c, *d_out;
    cudaMalloc(&d_c, N*sizeof(half2)); cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_c, h_c, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3m_hfma2_two_imm<<<(N+255)/256, 256>>>(d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float c_lo = __half2float(__low2half(h_c[i]));
        float o_lo = __half2float(__low2half(h_out[i]));
        if (fabsf(o_lo - (0.25f + c_lo)) > 0.05f) {
            printf("FAIL at %d: got %f exp %f\n", i, o_lo, 0.25f+c_lo); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
