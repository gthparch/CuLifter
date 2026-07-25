// DMMA.8x8x4 f64 GEMM fixture — MULTI-TILE (2x2 = 4 output 8x8 atom tiles, K=4).
//
// Plain GEMM D = alpha*A*Bᵀ + beta*C over a 16x16 output (4 atoms), the variant that
// EXPOSES the C/D m<->n square-tile transpose hazard (a single 8x8 atom self-conceals it;
// only off-diagonal output tiles (mi != ni) reveal a transposed C/D fragment map — exactly
// the second bug cutlass_realgemm_oracle root-caused in 19_tensorop_canonical). Re-audits
// that the warp-true atom + cute fragment map hold across all 4 tiles after the fix.
//
//   D[16][16],  A[16][4] row-major,  B[16][4] (=Bᵀ stored, col-major K x N),  C[16][16].
//   tile (mi,ni) in {0,1}^2 computes the 8x8 block D[mi*8 + m][ni*8 + n]:
//     D0[m][n] = sum_{k<4} A[mi*8+m][k]*B[ni*8+n][k]
//   epilogue: D = alpha*D0 + beta*C   (full output written).
// cute map per atom (lane g,t): a0=A[mi*8+g][t], b0=B[ni*8+g][t], d/c[v]=C[mi*8+g][ni*8+2t+v].
extern "C" __global__ void kern_gemm2x2(const double* A, const double* B,
                                        const double* C, double* D,
                                        double alpha, double beta)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;   // m/n within tile, in {0..7}
    const int t = lane & 3;    // k in {0..3}

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int ni = 0; ni < 2; ++ni) {
            double a0 = A[(mi * 8 + g) * 4 + t];        // A[mi*8+g][k=t]
            double b0 = B[(ni * 8 + g) * 4 + t];        // B[ni*8+g][k=t]   (col-major K x N)
            double c0 = C[(mi * 8 + g) * 16 + (ni * 8 + (t * 2 + 0))];
            double c1 = C[(mi * 8 + g) * 16 + (ni * 8 + (t * 2 + 1))];
            double z0 = 0.0, z1 = 0.0, d0, d1;
            asm volatile(
                "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
                : "=d"(d0), "=d"(d1)
                : "d"(a0), "d"(b0), "d"(z0), "d"(z1));
            D[(mi * 8 + g) * 16 + (ni * 8 + (t * 2 + 0))] = alpha * d0 + beta * c0;
            D[(mi * 8 + g) * 16 + (ni * 8 + (t * 2 + 1))] = alpha * d1 + beta * c1;
        }
    }
}
