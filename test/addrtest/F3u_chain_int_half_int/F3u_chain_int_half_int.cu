#include <cuda_fp16.h>

extern "C" __global__ void F3u_chain_int_half_int(const int* a, const int* b, const int* c,
    int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half ha = __int2half_rn(a[i]);
        half hb = __int2half_rn(b[i]);
        half hc = __int2half_rn(c[i]);
        half hr = __hfma(ha, hb, hc);
        out[i] = __half2int_rn(hr);
    }
}
