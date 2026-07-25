#include <cuda_fp16.h>

extern "C" __global__ void F3f_i2f_half(const int* in, half* out_signed,
    const unsigned int* in_u, half* out_unsigned, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_signed[i] = __int2half_rn(in[i]);
        out_unsigned[i] = __uint2half_rn(in_u[i]);
    }
}
