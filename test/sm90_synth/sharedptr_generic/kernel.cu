#include <cstdio>
#include <cuda_runtime.h>

// __shared__ T* pointers force the compiler to materialize GENERIC addresses
// of shared memory (SR_SWINHI window), store them in shared, reload, and
// dereference with generic LD.E — the quicksort lqsort_kernel pattern.
extern "C" __global__ void kern_sharedptr_generic(float* out, const float* in, int n, int flip) {
    __shared__ float a[64], b[64];
    __shared__ float *s, *sn;
    int tid = threadIdx.x;
    if (tid < n) { a[tid] = in[tid]; b[tid] = in[tid] * 2.0f; }
    __syncthreads();
    if (tid == 0) { s = flip ? b : a; sn = flip ? a : b; }
    __syncthreads();
    if (tid < n) out[tid] = s[tid] + 0.5f * sn[tid];
}

int main() {
    constexpr int N = 64;
    float hin[N], hout[N];
    for (int i = 0; i < N; i++) hin[i] = (float)(i + 1);
    float *din, *dout;
    cudaMalloc(&din, N * 4); cudaMalloc(&dout, N * 4);
    cudaMemcpy(din, hin, N * 4, cudaMemcpyHostToDevice);
    kern_sharedptr_generic<<<1, N>>>(dout, din, N, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(hout, dout, N * 4, cudaMemcpyDeviceToHost);
    FILE* f = fopen("ref_output.bin", "wb"); fwrite(hout, 4, N, f); fclose(f);
    // flip=1: s=b (2x), sn=a (1x) -> out = 2*in + 0.5*in = 2.5*in
    bool ok = true;
    for (int i = 0; i < N; i++) if (hout[i] != 2.5f * hin[i]) ok = false;
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    cudaFree(din); cudaFree(dout); return ok ? 0 : 1;
}
