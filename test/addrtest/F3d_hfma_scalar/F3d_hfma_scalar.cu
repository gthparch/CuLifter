#include <cuda_fp16.h>

extern "C" __global__ void F3d_hfma_scalar(const half* a, const half* b,
    const half* c, half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma(a[i], b[i], c[i]);
}
