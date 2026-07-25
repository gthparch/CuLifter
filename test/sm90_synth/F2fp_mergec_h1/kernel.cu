#include <cuda_fp8.h>
// Convert 4 floats into a packed fp8x4 (e4m3), one byte at a time.
// The compiler builds the i32 result via F2FP...PACK_AB_MERGE_C with the
// running register selected as Rd.H1 — exercising the high-half merge.
extern "C" __global__ void F2fp_mergec_h1(const float* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* p = in + i*4;
        __nv_fp8x4_e4m3 q(make_float4(p[0], p[1], p[2], p[3]));
        out[i] = *reinterpret_cast<unsigned int*>(&q);
    }
}
