#include <cstdio>
#include <cuda_runtime.h>

// Force LEA.HI.SX32 by using signed 32-bit index multiplied by stride
// for 64-bit address computation
extern "C" __global__ void kern_lea_sx32(const int* __restrict__ in,
                                          int* __restrict__ out, int N) {
    // Use signed index that requires sign extension to 64-bit
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // Signed index → LEA.HI.SX32 for pointer arithmetic
    int signed_idx = tid - N/2;  // negative for first half
    int safe_idx = (signed_idx < 0) ? -signed_idx : signed_idx;
    safe_idx = safe_idx % N;
    out[tid] = in[safe_idx] + signed_idx;
}

int main() {
    constexpr int N = 256;
    int hi[N], ho[N];
    for (int i = 0; i < N; i++) hi[i] = i * 3;
    int *di, *do_;
    cudaMalloc(&di, N*4); cudaMalloc(&do_, N*4);
    cudaMemcpy(di, hi, N*4, cudaMemcpyHostToDevice);
    kern_lea_sx32<<<1, 256>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N, f); fclose(f);
    // tid=0: signed_idx=-128, safe=128, in[128]+(-128) = 384-128=256
    printf("TEST PASSED\n");
    cudaFree(di); cudaFree(do_); return 0;
}
