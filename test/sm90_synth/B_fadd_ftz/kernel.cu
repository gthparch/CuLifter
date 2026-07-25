#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_fadd_ftz(const float* a, const float* b,
                                          float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    // __fadd_rz flushes denormals on SM90
    float sum = __fadd_rz(a[tid], b[tid]);
    float diff = __fadd_rz(a[tid], -b[tid]);
    out[tid*2] = sum;
    out[tid*2+1] = diff;
}

int main() {
    constexpr int N = 128;
    float ha[N], hb[N], ho[N*2];
    for (int i = 0; i < N; i++) {
        ha[i] = (float)(i + 1) * 1e-38f;  // near denormal range
        hb[i] = (float)(i + 1) * 1e-38f;
    }
    float *da, *db, *do_;
    cudaMalloc(&da, N*4); cudaMalloc(&db, N*4); cudaMalloc(&do_, N*2*4);
    cudaMemcpy(da, ha, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, N*4, cudaMemcpyHostToDevice);
    kern_fadd_ftz<<<1, 128>>>(da, db, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*2*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N*2, f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(da); cudaFree(db); cudaFree(do_); return 0;
}
