#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

// Force local memory (LDL/STL) by using large per-thread arrays
// that don't fit in registers. Use float4 to encourage LDL.128/STL.128.
extern "C" __global__ void kern_ldl_wide(const float* __restrict__ in,
                                          float* __restrict__ out, int N) {
    float local_buf[32];  // 128 bytes → spills to local memory
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    // Load into local array (STL)
    for (int j = 0; j < 8; j++)
        local_buf[j] = in[tid * 8 + j];

    // Process in local
    for (int j = 0; j < 8; j++)
        local_buf[j] = local_buf[j] * 2.0f + 1.0f;

    // Additional local usage to force spill
    for (int j = 8; j < 16; j++)
        local_buf[j] = local_buf[j-8] + (float)j;

    // Store back from local (LDL)
    float sum = 0.0f;
    for (int j = 0; j < 16; j++)
        sum += local_buf[j];
    out[tid] = sum;
}

int main() {
    constexpr int N = 128;
    float hi[N*8], ho[N];
    for (int i = 0; i < N*8; i++) hi[i] = (float)(i % 17 + 1) * 0.1f;
    float *di, *do_;
    cudaMalloc(&di, N*8*4); cudaMalloc(&do_, N*4);
    cudaMemcpy(di, hi, N*8*4, cudaMemcpyHostToDevice);
    kern_ldl_wide<<<1, 128>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N, f); fclose(f);
    bool ok = !isnanf(ho[0]) && !isinff(ho[0]) && ho[0] != 0.0f;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED (got %f)\n", ho[0]);
    cudaFree(di); cudaFree(do_); return ok ? 0 : 1;
}
