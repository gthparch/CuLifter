#include <cuda_fp16.h>

extern "C" __global__ void F3r_i2f_s16(const short* in, half* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __short2half_rn(in[i]);
}
