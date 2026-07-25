// Regression microbench for the half2 REGISTER-operand negation bug.
// cutlass LeftSiLUAndMul with ElementCompute=half lowers SiLu(x)=x*sigmoid(x)
// = x/(1+exp(-x)) through a half2 fast_exp path containing `HADD2.F32 R0,-RZ,
// -R5.H0_H0` (the -x for exp(-x)) and `HSETP2.EQ -R5.H0_H0,...` range-reduction.
// Before the fix, `_get_val_half2` / `lift_hsetp2` integer-negated the packed
// f16x2 word (e.g. -0x38003800 -> h0=-8.0 instead of -0.5), corrupting the
// sigmoid (it degenerated to a near-step function). The fix applies negation
// per-lane (fneg on <2 x half>).
#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/epilogue/thread/activation.h"
#include "left_silu_and_mul.h"

using OutputOp = cutlass::epilogue::thread::LeftSiLUAndMul<cutlass::half_t, 8,
    cutlass::half_t, cutlass::half_t, cutlass::FloatRoundStyle::round_to_nearest>;

extern "C" __global__ void silu_half2_neg(const cutlass::half_t* d0,
                                          const cutlass::half_t* d1,
                                          cutlass::half_t* d2, int n) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n) { OutputOp op{OutputOp::Params{}}; d2[i] = op(d0[i], d1[i]); }
}
