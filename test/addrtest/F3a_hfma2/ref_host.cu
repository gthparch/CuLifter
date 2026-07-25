#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3a_hfma2(const half2* a, const half2* b,
    const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(a[i], b[i], c[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    half2 h_a[N], h_b[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) {
        float v1 = (i%7+1)*0.5f, v2 = (i%5+1)*0.25f;
        h_a[i] = __halves2half2(__float2half(v1), __float2half(v1+0.1f));
        h_b[i] = __halves2half2(__float2half(v2), __float2half(v2+0.05f));
        h_c[i] = __halves2half2(__float2half(1.0f), __float2half(2.0f));
    }
    memset(h_out, 0, sizeof(h_out));
    half2 *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(half2)); cudaMalloc(&d_b, N*sizeof(half2));
    cudaMalloc(&d_c, N*sizeof(half2)); cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_a, h_a, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3a_hfma2<<<(N+255)/256, 256>>>(d_a, d_b, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float a_lo = __half2float(__low2half(h_a[i]));
        float b_lo = __half2float(__low2half(h_b[i]));
        float c_lo = __half2float(__low2half(h_c[i]));
        float o_lo = __half2float(__low2half(h_out[i]));
        if (fabsf(o_lo - (a_lo*b_lo+c_lo)) > 0.1f) {
            printf("FAIL at %d lo\n", i); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
