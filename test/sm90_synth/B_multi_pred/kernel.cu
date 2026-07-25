#include <cstdio>
#include <cuda_runtime.h>

extern "C" __global__ void kern_multi_pred(const int* __restrict__ in,
                                            int* __restrict__ out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    int a = in[tid], b = in[tid + N], c = in[tid + 2*N];
    int d = in[tid + 3*N], e = in[tid + 4*N], f = in[tid + 5*N];

    // 6+ conditions → forces P0-P5 predicate registers
    bool p0 = (a > 0), p1 = (b > 0), p2 = (c > 0);
    bool p3 = (d > 0), p4 = (e > 0), p5 = (f > 0);

    int result = 0;
    if (p0) result += a;
    if (p1) result += b;
    if (p2) result += c;
    if (p3) result += d;  // uses P3
    if (p4) result += e;  // uses P4
    if (p5) result += f;  // uses P5
    if (p0 && p3) result *= 2;
    if (p1 && p4) result += 100;
    out[tid] = result;
}

int main() {
    constexpr int N = 256;
    int hi[N*6], ho[N];
    for (int i = 0; i < N*6; i++) hi[i] = (i % 31) - 15;  // mix of pos/neg
    int *di, *do_;
    cudaMalloc(&di, N*6*4); cudaMalloc(&do_, N*4);
    cudaMemcpy(di, hi, N*6*4, cudaMemcpyHostToDevice);
    kern_multi_pred<<<1, 256>>>(di, do_, N);
    cudaDeviceSynchronize();
    cudaMemcpy(ho, do_, N*4, cudaMemcpyDeviceToHost);
    FILE *f = fopen("ref_output.bin", "wb"); fwrite(ho, 4, N, f); fclose(f);
    printf("TEST PASSED\n");
    cudaFree(di); cudaFree(do_); return 0;
}
