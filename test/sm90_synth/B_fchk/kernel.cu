#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_fchk(const float* __restrict__ in,
                                      int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float val = in[tid];
    // isfinite → FCHK instruction on SM90
    out[tid] = isfinite(val) ? 1 : 0;
}

int main() {
    constexpr int N = 256;
    float hi[N]; int ho[N];
    for (int i = 0; i < N; i++) hi[i] = (float)(i + 1) * 0.1f;
    hi[10] = INFINITY; hi[20] = NAN; hi[30] = -INFINITY;
    float *di; int *do_;
    cudaMalloc(&di, N*4); cudaMalloc(&do_, N*4);
    cudaMemcpy(di, hi, N*4, cudaMemcpyHostToDevice);
    kern_fchk<<<1, 256>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N, f); fclose(f);
    bool ok = (ho[0] == 1 && ho[10] == 0 && ho[20] == 0 && ho[30] == 0);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(di); cudaFree(do_); return ok ? 0 : 1;
}
