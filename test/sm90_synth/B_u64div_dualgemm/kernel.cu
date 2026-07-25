// Microbench: signed 64-bit divide + modulo by a RUNTIME 64-bit divisor.
//
// On sm90 ptxas lowers a runtime s64/s64 (and the matching s64%s64) to the
// 64-bit Granlund-Moller magic-reciprocal chain:
//   I2F.U64.RP -> MUFU.RCP -> VIADD +0x1FFFFFFE -> F2I.U64.TRUNC
//   -> a 64-bit Newton-Raphson reciprocal refinement
//   -> pre-correction remainder pair  rem = dividend - q*divisor
//   -> conditional-subtract correction + sign fixup.
// This is the same idiom 45_dual_gemm's grid-strided coordinate divide uses.
//
// Crucially, ptxas emits the pre-correction remainder HIGH word in TWO
// interchangeable opcode forms:
//   IMAD.X  rem_hi, Dhi, 0x1, ~hi(q*d), P        (the form B_u64mod hits)
//   IADD3.X rem_hi, ~hi(q*d), Dhi, RZ, P, !PT    (the form 45_dual_gemm hits)
// Both compute Dhi - hi(q*d) - borrow.  IntDivRecognize._try_rewrite_u64
// originally recognized only the IMAD.X form, so the IADD3.X chain was lifted
// FAITHFULLY (software rcp_f32, whose bits differ from the HW SFU) and produced
// a wrong / off-by-a-few-multiples 64-bit quotient & remainder.  A single
// s64 div+mod kernel exercises BOTH forms (the divide chain -> IMAD.X rem_hi,
// the modulo chain -> IADD3.X rem_hi), so it pins the gap precisely.
#include <cstdint>

extern "C" __global__ void kern_s64div(long long* q, long long* r,
                                       const long long* a, const long long* b,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    q[i] = a[i] / b[i];
    r[i] = a[i] % b[i];
}
