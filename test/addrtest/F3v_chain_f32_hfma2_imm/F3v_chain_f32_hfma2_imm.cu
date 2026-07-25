#include <cuda_fp16.h>

extern "C" __global__ void F3v_chain_f32_hfma2_imm(const float* in_a, const float* in_b,
    float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half2 h = __floats2half2_rn(in_a[i], in_b[i]);
        half2 r = __hfma2(h, __float2half2_rn(0.5f), __float2half2_rn(1.0f));
        out[i*2]   = __low2float(r);
        out[i*2+1] = __high2float(r);
    }
}
