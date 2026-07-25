#include <cuda_fp16.h>

extern "C" __global__ void F3b_hadd2(const half2* a, const half2* b,
    half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hadd2(a[i], b[i]);
}
