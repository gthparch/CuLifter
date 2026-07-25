#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

extern "C" __global__ void F3v_chain_f32_hfma2_imm(const float* in_a, const float* in_b,
    float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half2 h = __floats2half2_rn(in_a[i], in_b[i]);
        half2 r = __hfma2(h, __float2half2_rn(0.5f), __float2half2_rn(1.0f));
        out[i*2]   = __low2float(r);
        out[i*2+1] = __high2float(r);
    }
}

int main(int argc, char** argv) {
    constexpr int N = 512;
    float h_a[N], h_b[N], h_out[N*2];
    for (int i = 0; i < N; i++) { h_a[i]=(i%7+1)*0.4f; h_b[i]=(i%5+1)*0.6f; }
    memset(h_out, 0, sizeof(h_out));
    float *d_a, *d_b, *d_out;
    cudaMalloc(&d_a, N*sizeof(float)); cudaMalloc(&d_b, N*sizeof(float));
    cudaMalloc(&d_out, N*2*sizeof(float));
    cudaMemcpy(d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, N*2*sizeof(float));
    F3v_chain_f32_hfma2_imm<<<(N+255)/256, 256>>>(d_a, d_b, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N*2*sizeof(float), cudaMemcpyDeviceToHost);
    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb"); fwrite(h_out, sizeof(float), N*2, f); fclose(f);
    }
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float ha = __half2float(__float2half(h_a[i]));
        float exp_lo = __half2float(__float2half(ha*0.5f+1.0f));
        if (fabsf(h_out[i*2]-exp_lo)>0.05f) { printf("FAIL at %d\n",i); ok=false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_out);
    return ok ? 0 : 1;
}
