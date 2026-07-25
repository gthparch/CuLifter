// DMMA.8x8x4 f64 TRMM fixture (one warp, K=8 => TWO m8n8k4 atoms accumulated;
// A is LOWER-triangular = left-side trmm operand).
//
// Triangular matrix multiply (TRMM), left-side / lower-fill / non-unit diag,
// exactly as 32_basic_trmm's ReferenceTrmm_kernel computes:
//   C[i][j] = alpha * sum_{k=0..M-1} A[i][k]*B[k][j]          (A lower-tri: A[i][k]=0 for i<k)
// Output is FULL (every (i,j) written); the triangularity is in the A OPERAND, not the
// write-set. The unused strict-upper triangle of A is zero, so its products vanish — this
// audits the DMMA atom on a triangular operand + 2-K-group accumulation (K=8 = 2x k4).
//
// cute MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> mapping per atom (kg = K-group, k = kg*4 + t):
//   a0     = A[m=g][k]                          (A is M8 x K8, row-major; lower-tri)
//   b0     = Bcm[n=g][k] = B[k][n]              (B operand col-major K x N => stored N x K)
//   accumulate over kg in {0,1}; C-operand carries the running accumulator across K-groups.
//   d/c[v] = D[m=g][n=2t+v]   v in {0,1}        (row-major M8 x N8)
// Epilogue: D = alpha * accum   (beta=0 for trmm; C is the destination only).
extern "C" __global__ void kern_trmm(const double* A, const double* Bcm, double* D,
                                     double alpha)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;   // m (for A) / n (for B) in {0..7}
    const int t = lane & 3;    // within-group k in {0..3}
    double acc0 = 0.0, acc1 = 0.0;   // running accumulators (the 2 D frags this lane owns)

#pragma unroll
    for (int kg = 0; kg < 2; ++kg) {
        const int k = kg * 4 + t;                 // global K index for this lane's frag
        double a0 = A[g * 8 + k];                 // A[m=g][k]   (A is M8 x K8 row-major)
        double b0 = Bcm[g * 8 + k];               // Bcm[n=g][k] = B[k][n]  (col-major)
        double d0, d1;
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
            "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
            : "=d"(d0), "=d"(d1)
            : "d"(a0), "d"(b0), "d"(acc0), "d"(acc1));   // accumulate into running acc
        acc0 = d0; acc1 = d1;
    }

#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int idx = (g * 8) + (t * 2 + v);
        D[idx] = alpha * (v == 0 ? acc0 : acc1);     // full output; beta=0
    }
}
