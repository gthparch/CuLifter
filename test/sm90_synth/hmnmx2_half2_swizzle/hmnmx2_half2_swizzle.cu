// Regression microbench for the half2 .H0_H0 SWIZZLE being dropped on HMNMX2.
//
// `__hmax2(x, __half2half2(s))` lowers (sm_90) to
//     LDG.E.U16 R5, [s]          ; R5 = {s, 0x0000}  (16-bit zero-extended load)
//     HMNMX2    R9, R2, R5.H0_H0, !PT
// The `.H0_H0` swizzle broadcasts R5's LOW half (s) to BOTH lanes, so the GPU
// computes {max(x.lo, s), max(x.hi, s)}.
//
// Before the generic half2-operand-modifier fix, `lift_hmnmx2` fetched its
// register operands via the GENERIC `_get_val` + raw bitcast to <2 x half>,
// DROPPING the .H0_H0 swizzle: it used R5 verbatim = {s, 0.0}, computing
// {max(x.lo, s), max(x.hi, 0.0)} — the HIGH lane is wrong whenever s != 0.
// The fix routes operands through `_get_val_half2`, which honors the swizzle
// (broadcast lane0). This is the exact pattern seen 32x in CB_hgemm and the
// cutlass 13_fused_two_gemms/convs SASS.
//
// (HMNMX2 also gets per-lane `-Rx`/`|Rx|` for free via _get_val_half2, though
// the corpus only exhibits the swizzle, not negation, for this op.)
#include <cuda_fp16.h>

extern "C" __global__ void hmnmx2_half2_swizzle(const __half2* a, const __half* s,
                                                __half2* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    __half2 x = a[i];
    __half  sc = s[i];
    out[i] = __hmax2(x, __half2half2(sc));   // {max(x.lo,sc), max(x.hi,sc)}
  }
}
