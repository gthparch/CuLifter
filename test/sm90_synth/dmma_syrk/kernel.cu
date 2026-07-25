// DMMA.8x8x4 f64 SYRK fixture (one warp, single m8n8k4 atom; B := A => A·Aᵀ).
//
// Symmetric rank-k update (SYRK), lower-fill, exactly as 31_basic_syrk's
// ReferenceSyrk_kernel computes (just hand-rolled on the DMMA.8x8x4 atom instead
// of the CUTLASS RankKUniversal device pipeline — whose 43-field Params struct is
// out of scope; see NOTES). Math (lower triangle i>=j only):
//   D[i][j] = alpha * sum_k A[i][k]*A[j][k] + beta * C[i][j]      (i >= j)
//   D[i][j] = canary                                              (i <  j)   <-- UNwritten
//
// The atom feeds B := A so the GEMM core D0[m][n] = sum_k A[m][k]*A[n][k] IS A·Aᵀ
// (the symmetric kernel of SYRK). Each lane uses the cute
// MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> mapping (same as test/sm90_synth/dmma884):
//   a0     = A[m=g][k=t]                     (A is M8 x K4, row-major)
//   bA0    = A[n=g][k=t]                     (B operand := A, same row-major buf)
//   c/d[v] = C[m=g][n=2t+v]    v in {0,1}
// then writes ONLY the lower triangle (n<=m): epilogue D = alpha*AB + beta*C; the
// upper triangle (n>m) is left untouched (canary) to audit the triangular write-set.
extern "C" __global__ void kern_syrk(const double* A, const double* C, double* D,
                                     double alpha, double beta)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;   // t/4  in {0..7}  -> m index for a0, n index for bA0
    const int t = lane & 3;    // t%4  in {0..3}  -> k
    double a0, bA0, c[2], d[2];
    a0  = A[g * 4 + t];                 // A[m=g][k=t]
    bA0 = A[g * 4 + t];                 // B := A   -> A[n=g][k=t]
#pragma unroll
    for (int v = 0; v < 2; ++v) c[v] = C[(g * 8) + (t * 2 + v)];   // C[m=g][n=2t+v]

    // pure A·Aᵀ core (C-operand cleared to 0 so the atom returns the GEMM product
    // alone; the alpha/beta epilogue is applied in the store below — mirrors
    // CUTLASS: accum.clear() then alpha*accum + beta*source).
    double z0 = 0.0, z1 = 0.0;
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=d"(d[0]), "=d"(d[1])
        : "d"(a0), "d"(bA0), "d"(z0), "d"(z1));

#pragma unroll
    for (int v = 0; v < 2; ++v) {
        const int m = g, n = t * 2 + v;
        if (n <= m) {                                  // LOWER triangle (incl diag)
            D[(g * 8) + (t * 2 + v)] = alpha * d[v] + beta * c[v];
        }
        // n > m: leave D untouched (host pre-fills canary, must survive)
    }
}
