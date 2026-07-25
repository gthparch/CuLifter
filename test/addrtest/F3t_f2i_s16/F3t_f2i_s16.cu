#include <cuda_fp16.h>

extern "C" __global__ void F3t_f2i_s16(const half* in, short* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __half2short_rn(in[i]);
}
