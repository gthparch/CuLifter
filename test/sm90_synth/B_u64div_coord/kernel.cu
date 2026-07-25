// Microbench: EXACT 45_dual_gemm shape — a signed 64-bit magic-reciprocal
// DIVIDE (I2F.U64.RP → MUFU.RCP → VIADD +0x1FFFFFFE → F2I.U64.TRUNC + 64-bit
// Newton-Raphson) whose QUOTIENT is used ONLY as a 64-bit address index
// (coord*stride+base), like dual_gemm's grid-strided `coord = index/N`.
//
// That codegen (quotient -> address, no direct store of the quotient) is the
// case that makes ptxas schedule the pre-correction remainder HIGH word as
//   IADD3.X rem_hi, ~hi(q*d), Dhi, RZ, P, !PT
// (vs the IMAD.X form a direct `q = a/b` store emits).  IntDivRecognize must:
//   (1) recognise that IADD3.X rem-hi form (_match_rem_hi), AND
//   (2) recover the QUOTIENT as UDIV64/UDIV64HI (_find_precorr_quotient_pair),
// because for a DIVIDE the remainder rewrite only SUPPRESSES the Granlund-
// Möller +1 correction — it locks in the faithful (software-rcp, wrong) NR
// quotient estimate unless the pre-correction quotient is itself made exact.
//
// The kernel writes a[i] at out[coord]; the host picks (a[i], b[i]) so every
// coord = a[i]/b[i] is distinct, non-negative and in-bounds, then verifies
// out[coord_native] == a[i].  A wrong quotient writes a[i] to the wrong slot,
// so the expected slot is left untouched (sentinel) => VALUE mismatch, never a
// segfault (stride pins the write inside the buffer).
#include <cstdint>

extern "C" __global__ void kern_coord_div(long long* out, const long long* a,
                                          const long long* b, long long stride,
                                          int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    long long coord = a[i] / b[i];   // s64 magic-reciprocal divide
    out[coord * stride] = a[i];      // quotient used ONLY as an address index
}
