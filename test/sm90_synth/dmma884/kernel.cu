// DMMA.8x8x4 f64 tensor-core MMA fixture (one warp, single m8n8k4 atom).
//
// Each lane loads its A/B/C fragments from plain buffers using the CUTLASS cute
// MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> mapping, performs ONE mma.sync.m8n8k4,
// and scatters its 2 D fragments back:
//   A: SM80_8x4    -> a[0]      = A[m=t/4][k=t%4]            (A is M8 x K4, row-major)
//   B: SM80_8x4    -> b[0]      = B[n=t/4][k=t%4]            (B is N8 x K4, i.e. col-major K x N)
//   C/D: SM80_8x8_Row -> c/d[v] = C[m=t/4][n=(t%4)*2+v]     (v in {0,1}, C/D M8 x N8 row-major)
//     (cute column-major codomain; a single 8x8 atom cannot distinguish the C/D
//      m<->n convention, but the full multi-tile GEMM in 19_tensorop_canonical does.)
// Result: D[m][n] = C[m][n] + sum_k A[m][k]*B[n][k].
//
// This isolates the DMMA.8x8x4 atom — the path 19_tensorop_canonical (canonical f64
// tensorop GEMM) exercises and that the lifter previously stubbed as a per-lane scalar FMA.
extern "C" __global__ void kern_dmma884(const double* A, const double* B,
                                        const double* C, double* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;   // t/4  in {0..7}
    const int t = lane & 3;    // t%4  in {0..3}
    double a0, b0, c[2], d[2];
    a0 = A[g * 4 + t];                 // A[m=g][k=t]
    b0 = B[g * 4 + t];                 // B[n=g][k=t]
#pragma unroll
    for (int v = 0; v < 2; ++v) c[v] = C[(g * 8) + (t * 2 + v)];   // C[m=g][n=2t+v]

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{%0, %1}, {%2}, {%3}, {%4, %5};\n"
        : "=d"(d[0]), "=d"(d[1])
        : "d"(a0), "d"(b0), "d"(c[0]), "d"(c[1]));

#pragma unroll
    for (int v = 0; v < 2; ++v) D[(g * 8) + (t * 2 + v)] = d[v];   // D[m=g][n=2t+v]
}
