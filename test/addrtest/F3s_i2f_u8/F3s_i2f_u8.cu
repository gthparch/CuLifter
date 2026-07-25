#include <cuda_fp16.h>

extern "C" __global__ void F3s_i2f_u8(const unsigned char* in, half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __ushort2half_rn((unsigned short)in[i]);
}
