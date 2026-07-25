#include <cuda_fp8.h>
// fp8x4 e5m2 pack from 4 floats (exercises E5M2 PACK_AB_MERGE_C .H1).
extern "C" __global__ void F2fp_mergec_h1_e5m2(const float* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* p = in + i*4;
        __nv_fp8x4_e5m2 q(make_float4(p[0], p[1], p[2], p[3]));
        out[i] = *reinterpret_cast<unsigned int*>(&q);
    }
}
