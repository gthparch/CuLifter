#include <cuda_fp8.h>
// Build an fp8x4 by converting each float separately and packing - aims to
// reproduce the F2FP MERGE_C Rd, RZ, Rb, Rd.H1 (A=RZ) shift-register idiom
// seen in 88_hopper_fmha_fp8.
extern "C" __global__ void F2fp_mergec_rz(const float* in, unsigned int* out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* p = in + i*4;
        unsigned int r = 0;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            __nv_fp8_e4m3 q(p[k]);
            unsigned char byte = *reinterpret_cast<unsigned char*>(&q);
            r |= ((unsigned int)byte) << (8*k);
        }
        out[i] = r;
    }
}
