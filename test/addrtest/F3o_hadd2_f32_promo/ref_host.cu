#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3o_hadd2_f32_promo(const half2* a, const half2* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half2 s = __hadd2(a[i], b[i]);
        out[i*2]   = __low2float(s);
        out[i*2+1] = __high2float(s);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    half2 h_a[N], h_b[N]; float h_out[N*2];
    for (int i = 0; i < N; i++) {
        h_a[i] = __halves2half2(__float2half((i%7+1)*0.3f), __float2half((i%5+1)*0.4f));
        h_b[i] = __halves2half2(__float2half((i%9+1)*0.2f), __float2half((i%3+1)*0.6f));
    }
    memset(h_out, 0, sizeof(h_out));
    half2 *d_a, *d_b; float *d_out;
    cudaMalloc(&d_a, N*sizeof(half2)); cudaMalloc(&d_b, N*sizeof(half2));
    cudaMalloc(&d_out, N*2*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(half2), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*2*sizeof(float));
    F3o_hadd2_f32_promo<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*2*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N*2, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float a_lo = __half2float(__low2half(h_a[i])), b_lo = __half2float(__low2half(h_b[i]));
        float exp_lo = __half2float(__float2half(a_lo+b_lo));
        if (fabsf(h_out[i*2] - exp_lo) > 0.01f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
