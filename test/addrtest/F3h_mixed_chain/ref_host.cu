#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3h_mixed_chain(const float* a, const float* b,
    const float* c, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half ha = __float2half(a[i]);
        half hb = __float2half(b[i]);
        half hc = __float2half(c[i]);
        half hr = __hfma(ha, hb, hc);
        out[i] = __half2float(hr);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 1024;
    float h_a[N], h_b[N], h_c[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (i%7+1)*0.5f; h_b[i] = (i%5+1)*0.25f; h_c[i] = 1.0f;
    }
    memset(h_out, 0, sizeof(h_out));
    float *d_a, *d_b, *d_c, *d_out;
    cudaMalloc(&d_a, N*sizeof(float)); cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_c, N*sizeof(float)); cudaMalloc(&d_out, N*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*sizeof(float));
    F3h_mixed_chain<<<(N+255)/256, 256>>>(d_a, d_b, d_c, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        // Simulate fp16 chain on host: float→half→float for each operand, then fma in fp32
        float ha = __half2float(__float2half(h_a[i]));
        float hb = __half2float(__float2half(h_b[i]));
        float hc = __half2float(__float2half(h_c[i]));
        float exp = __half2float(__float2half(ha * hb + hc));
        if (fabsf(h_out[i] - exp) > 0.05f) {
            printf("FAIL at %d: got %f exp %f\n", i, h_out[i], exp); ok=false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_out);
    return ok ? 0 : 1;
}
