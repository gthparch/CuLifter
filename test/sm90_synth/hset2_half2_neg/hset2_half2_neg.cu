// Regression microbench for the half2 REGISTER-operand negation bug on HSET2.
//
// `__heq2(__hneg2(x), y)` lowers (sm_90) to a single
//     HSET2.BF.EQ.AND Rd, -Ra, Rb, PT
// i.e. a packed-f16x2 equality with a NEGATED register operand, whose result
// is the per-lane boolean mask 0xFFFF / 0x0000.
//
// Before the generic half2-operand-modifier fix, `lift_hset2` fetched its
// register operands via the GENERIC `_get_val` (raw bitcast to <2 x half>),
// so the `-Ra` modifier was applied as an i32 two's-complement negate of the
// whole packed word (e.g. -0x38003800 -> 0xC7FFC800, giving h0 = -8.0 instead
// of -0.5) AND the per-lane semantics were lost. The compare then used the
// wrong lane values => wrong mask. The fix routes ra/rb through
// `_get_val_half2`, applying `-Rx` PER-LANE (fneg on <2 x half>).
//
// The output mask is stored verbatim, so the host can check the exact bits.
#include <cuda_fp16.h>

extern "C" __global__ void hset2_half2_neg(const __half2* a, const __half2* b,
                                           __half2* out, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) {
    __half2 x = a[i];
    __half2 y = b[i];
    out[i] = __heq2(__hneg2(x), y);   // (-x == y) ? 0xFFFF : 0x0000 per lane
  }
}
