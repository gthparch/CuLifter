#include <cuda_fp16.h>

extern "C" __global__ void F3g_f2i_half(const half* in, int* out_signed,
    unsigned int* out_unsigned, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        out_signed[i] = __half2int_rn(in[i]);
        out_unsigned[i] = __half2uint_rn(in[i]);
    }
}
