#include <cuda_fp16.h>

extern "C" __global__ void F3a_hfma2(const half2* a, const half2* b,
    const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(a[i], b[i], c[i]);
}
