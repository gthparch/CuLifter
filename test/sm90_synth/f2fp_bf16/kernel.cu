#include <cstdio>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

extern "C" __global__ void kern_f2fp_bf16(const float* __restrict__ in,
                                           __nv_bfloat162* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = in[tid * 2], b = in[tid * 2 + 1];
    out[tid] = __float22bfloat162_rn(make_float2(a, b));
}

int main() {
    constexpr int N = 512;
    float h_in[N * 2];
    for (int i = 0; i < N * 2; i++) h_in[i] = (float)(i % 31 + 1) * 0.1f;
    float *d_in; __nv_bfloat162 *d_out;
    cudaMalloc(&d_in, N * 2 * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(__nv_bfloat162));
    cudaMemcpy(d_in, h_in, N * 2 * sizeof(float), cudaMemcpyHostToDevice);
    kern_f2fp_bf16<<<(N + 255) / 256, 256>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    __nv_bfloat162 h_out[N];
    cudaMemcpy(h_out, d_out, N * sizeof(__nv_bfloat162), cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_out, sizeof(__nv_bfloat162), N, f); fclose(f);
    bool ok = true;
    for (int i = 0; i < 10; i++) {
        float exp_val = h_in[i * 2];
        float got = __bfloat162float(((__nv_bfloat16*)h_out)[i * 2]);
        if (fabsf(got - exp_val) > 0.1f) { ok = false; break; }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(d_in); cudaFree(d_out);
    return ok ? 0 : 1;
}
