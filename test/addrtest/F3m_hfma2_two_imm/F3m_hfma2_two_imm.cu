#include <cuda_fp16.h>

extern "C" __global__ void F3m_hfma2_two_imm(const half2* c, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __hfma2(__float2half2_rn(0.5f), __float2half2_rn(0.5f), c[i]);
}
