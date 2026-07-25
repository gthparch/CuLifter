#include <cuda_fp16.h>

extern "C" __global__ void F3i_hfma2_imm_half(const half2* a, const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(a[i], __float2half2_rn(0.5f), c[i]);
}
