#include <cuda_fp16.h>

extern "C" __global__ void F3o_hadd2_f32_promo(const half2* a, const half2* b, float* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        half2 s = __hadd2(a[i], b[i]);
        out[i*2]   = __low2float(s);
        out[i*2+1] = __high2float(s);
    }
}
