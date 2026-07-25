#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_ffma_round(const float* a, const float* b,
                                            float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // Different rounding modes
    float rz = __fmaf_rz(a[tid], b[tid], 0.0f);  // FFMA.RZ
    float rn = __fmaf_rn(a[tid], b[tid], 0.0f);   // FFMA (default RN)
    float rd = __fmaf_rd(a[tid], b[tid], 0.0f);   // FFMA.RM (round down)
    float ru = __fmaf_ru(a[tid], b[tid], 0.0f);   // FFMA.RP (round up)
    out[tid*4+0] = rz;
    out[tid*4+1] = rn;
    out[tid*4+2] = rd;
    out[tid*4+3] = ru;
}

int main() {
    constexpr int N = 64;
    float ha[N], hb[N], ho[N*4];
    for (int i = 0; i < N; i++) { ha[i] = 1.0000001f; hb[i] = 1.0000001f; }
    float *da, *db, *do_;
    cudaMalloc(&da, N*4); cudaMalloc(&db, N*4); cudaMalloc(&do_, N*4*4);
    cudaMemcpy(da, ha, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N*4, cudaMemcpyHostToDevice);
    kern_ffma_round<<<1, 64>>>(da, db, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N*4, f); fclose(f);
    bool ok = !isnanf(ho[0]) && !isnanf(ho[1]) && !isnanf(ho[2]) && !isnanf(ho[3]);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(da); cudaFree(db); cudaFree(do_); return ok ? 0 : 1;
}
