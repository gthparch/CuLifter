// MINIMAL value-level repro of an IntDivRecognize 64-bit-divide gap, reduced from
// the 42_ampere_tensorop_group_conv im2col activation-gather addressing divergence.
//
// A 64-bit (long) flat thread id divided AND mod'd by a RUNTIME int divisor, with
// the quotient/remainder also feeding a gather, makes ptxas lower the op into TWO
// separate magic-divide chains: a 64-bit divide-helper subroutine (CALL) PLUS a
// 32-bit inline magic-divide fast path. IntDivRecognize rewrites the 64-bit-helper
// leaf (UREM64) and SOME 32-bit fast-path leaves, but LEAVES at least one 32-bit
// fast-path magic-divide leaf as a faithful rcp_f32 SOFTWARE-reciprocal chain.
// On the x86 backend rcp_f32 is an approximation -> the unrecognized leaf yields a
// WRONG quotient/remainder (t=9, D=18: q=3 instead of 0, r=-45 instead of 9).
//
// The gather index is CLAMPED so the divergence shows as a VALUE mismatch (q/r/go)
// rather than an OOB crash. The store index is a clean 32-bit i. This is the conv
// im2col npq-decompose (npq_row = tid/CRS with tid 64-bit) that breaks
// 42_ampere_tensorop_group_conv.
extern "C" __global__ void kern_div64_runtime_gather(
    const float* __restrict__ src, float* __restrict__ go,
    int* __restrict__ out_q, int* __restrict__ out_r,
    int total, int D, int SRC)
{
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;   // 64-bit id
    if (tid >= (long)total) return;
    const int i = (int)tid;                       // clean 32-bit store index
    long qo = tid / D, rem = tid % D;             // 64-bit / runtime divisor
    out_q[i] = (int)qo;                           // expected tid/D
    out_r[i] = (int)rem;                          // expected tid%D
    long idx64 = qo * D + rem;                    // should == tid
    int idx = (int)(idx64 % SRC);
    if (idx < 0) idx = 0; if (idx >= SRC) idx = SRC-1;   // clamp -> no OOB crash
    go[i] = src[idx];                             // expected src[tid % SRC]
}
