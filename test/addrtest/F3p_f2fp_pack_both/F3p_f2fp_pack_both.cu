#include <cuda_fp16.h>

extern "C" __global__ void F3p_f2fp_pack_both(const float* a, const float* b, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        out[i] = __floats2half2_rn(a[i], b[i]);
}
