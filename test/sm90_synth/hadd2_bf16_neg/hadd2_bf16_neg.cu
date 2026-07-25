// BF16-variant lock for the half2 per-lane register-negation path.
//
// `__hadd2(__hneg2(x), y)` on __nv_bfloat162 lowers (sm_90) to:
//     HFMA2.BF16_V2 R0, -RZ.H0_H0, RZ.H0_H0, -R2   ; R0 = -(RZ*RZ) + (-x) = -x
//     HADD2.BF16_V2 R9, R0, R4                      ; (-x) + y
// i.e. the per-lane negate is realized as a NEGATED accumulator operand (`-R2`)
// on HFMA2.BF16_V2. This exercises the bf16 branch of `_get_val_half2`
// (is_bf16=True -> <2 x bfloat>, fneg = per-lane sign-bit flip; `_v2_fabs`
// masks 0x7FFF7FFF which is also the bf16 per-lane sign bit).
//
// Confirms the per-lane neg/abs fix is correct for bfloat16, not just fp16.
#include <cuda_bf16.h>

extern "C" __global__ void hadd2_bf16_neg(const __nv_bfloat162* a,
                                          const __nv_bfloat162* b,
                                          __nv_bfloat162* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    __nv_bfloat162 x = a[i];
    __nv_bfloat162 y = b[i];
    out[i] = __hadd2(__hneg2(x), y);   // (-x) + y per lane
  }
}
