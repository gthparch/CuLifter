#include <cuda_fp16.h>

extern "C" __global__ void F3q_f2fp_single_slot(const float* in, half2* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half h = __float2half(in[i]);
        out[i] = __halves2half2(h, __float2half(0.0f));
    }
}
