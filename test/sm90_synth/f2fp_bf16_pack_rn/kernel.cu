// BF16 PACK-convert truncation/rounding audit.
// out[i] = pack two consecutive f32 -> bf16x2 via round-to-nearest-even.
// This is the EXACT cutlass mainloop / epilogue op: F2FP.BF16.F32.PACK_AB
// (NumericConverter<bfloat16_t, float, FloatRoundStyle::round_to_nearest>).
// __float22bfloat162_rn lowers to F2FP.BF16.F32.PACK_AB (no .RZ) per ptxas.
// The INPUTS are chosen to stress RN-even boundaries: exact ties (mantissa
// bit17==0/1 with the dropped bits == 0x8000) decide round-up vs round-to-even.
#include <cuda_bf16.h>
extern "C" __global__ void kern_f2fp_bf16_pack(const float* __restrict__ in,
                                               __nv_bfloat162* __restrict__ out,
                                               int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    float a = in[tid * 2], b = in[tid * 2 + 1];
    out[tid] = __float22bfloat162_rn(make_float2(a, b));
}
