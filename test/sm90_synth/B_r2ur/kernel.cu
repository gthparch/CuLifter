#include <cstdio>
#include <cuda_runtime.h>

// R2UR is generated when compiler promotes a per-thread value to uniform
// Use __shfl_sync to create a warp-uniform value from thread 0
extern "C" __global__ void kern_r2ur(const int* __restrict__ in,
                                      int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int val = in[0];  // all threads read same address → compiler may use R2UR
    // Force the value through per-thread computation then back to uniform
    val = __shfl_sync(0xffffffff, val + threadIdx.x, 0);  // broadcast lane 0
    out[tid] = val + tid;
}

int main() {
    constexpr int N = 256;
    int hi[N], ho[N];
    for (int i = 0; i < N; i++) hi[i] = 100;
    int *di, *do_;
    cudaMalloc(&di, N*4); cudaMalloc(&do_, N*4);
    cudaMemcpy(di, hi, N*4, cudaMemcpyHostToDevice);
    kern_r2ur<<<1, 256>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N, f); fclose(f);
    // val=100+0=100 (from lane 0), out[0]=100+0=100
    bool ok = (ho[0] == 100);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED (got %d)\n", ho[0]);
    cudaFree(di); cudaFree(do_); return ok ? 0 : 1;
}
