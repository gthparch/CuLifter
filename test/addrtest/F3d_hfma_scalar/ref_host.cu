#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3d_hfma_scalar(const half* a, const half* b,
    const half* c, half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma(a[i], b[i], c[i]);
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    half h_a[N], h_b[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = __float2half((i%7+1)*0.5f);
        h_b[i] = __float2half((i%5+1)*0.25f);
        h_c[i] = __float2half(1.0f);
    }
    memset(h_out, 0, sizeof(h_out));
    half *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(half)); cudaMalloc(&d_b, N*sizeof(half));
    cudaMalloc(&d_c, N*sizeof(half)); cudaMalloc(&d_out, N*sizeof(half));
    cudaMemcpy(d_a, h_a, N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(half));
    F3d_hfma_scalar<<<(N+255)/256, 256>>>(d_a, d_b, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(half), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(half), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float a = __half2float(h_a[i]), b = __half2float(h_b[i]);
        float c = __half2float(h_c[i]), o = __half2float(h_out[i]);
        if (fabsf(o - (a*b+c)) > 0.1f) {
            printf("FAIL at %d\n", i); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
