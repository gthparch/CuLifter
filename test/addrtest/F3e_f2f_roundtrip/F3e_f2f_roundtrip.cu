#include <cuda_fp16.h>

extern "C" __global__ void F3e_f2f_roundtrip(const float* in, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half h = __float2half(in[i]);
        out[i] = __half2float(h);
    }
}
