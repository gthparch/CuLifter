// Warp-synchronous blocked forward substitution (cuBLAS kernel_trsm_l_mul32
// idiom, CB_strsm): per row r, lane r alone computes the quotient (divergent
// diamond with an fdiv -> FCHK/MUFU.RCP slow-path pattern), then SHFL.IDX
// broadcasts it and lanes > r accumulate.  The unrolled loop puts a shfl
// after every diamond — exactly the shape clang -O2 tail-duplicates in the
// lifted IR.  Guards the collective-site keying: if warp groups are keyed by
// binary call site instead of the lifted site argument, the duplicated
// clones split each warp into {lane r} + {others} and every lane keeps its
// own value (x == alpha*b/diag instead of the solve).
extern "C" __global__ void kern_shflsubst(const float* A, const float* b,
                                          float* x, float alpha)
{
    const int lane = threadIdx.x & 31;
    const int col  = threadIdx.y;           // one warp per RHS column
    float acc = 0.0f;                       // sum_{k<lane} A(lane,k)*x_k
    const float bv = b[col * 32 + lane];
    float out = 0.0f;
#pragma unroll
    for (int r = 0; r < 32; ++r) {
        if (lane == r)
            out = (alpha * bv - acc) / A[r * 32 + r];  // A col-major: (r,r)
        const float xr = __shfl_sync(0xffffffffu, out, r);
        if (lane > r)
            acc += xr * A[r * 32 + lane];              // A(lane, r)
    }
    x[col * 32 + lane] = out;
}
