#include <cuda_fp16.h>

extern "C" __global__ void F3h_mixed_chain(const float* a, const float* b,
    const float* c, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half ha = __float2half(a[i]);
        half hb = __float2half(b[i]);
        half hc = __float2half(c[i]);
        half hr = __hfma(ha, hb, hc);
        out[i] = __half2float(hr);
    }
}
