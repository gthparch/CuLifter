#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3b_hadd2(const half2* a, const half2* b,
    half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hadd2(a[i], b[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    half2 h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        float v1 = (i%11+1)*0.3f, v2 = (i%9+1)*0.7f;
        h_a[i] = __halves2half2(__float2half(v1), __float2half(-v1));
        h_b[i] = __halves2half2(__float2half(v2), __float2half(v2));
    }
    memset(h_out, 0, sizeof(h_out));
    half2 *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(half2)); cudaMalloc(&d_b, N*sizeof(half2));
    cudaMalloc(&d_out, N*sizeof(half2));
    cudaMemcpy(d_a, h_a, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half2));
    F3b_hadd2<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half2), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half2), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float a_lo = __half2float(__low2half(h_a[i]));
        float b_lo = __half2float(__low2half(h_b[i]));
        float o_lo = __half2float(__low2half(h_out[i]));
        if (fabsf(o_lo - (a_lo+b_lo)) > 0.05f) {
            printf("FAIL at %d lo\n", i); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
