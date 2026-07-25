// DMMA.16x8x16 f64 tensor-core MMA fixture (one warp).
//
// Each lane loads its A/B/C fragments from plain row-major buffers using the
// CUTLASS cute MMA_Traits<SM90_16x8x16_F64F64F64F64_TN> mapping, performs one
// mma.sync, and scatters its D fragments back:
//   lane = 4*g + t
//   a[v0+2*v1] = A[g + 8*v0][t + 4*v1]
//   b[v]       = B[t + 4*v][g]          (B stored row-major K x N)
//   c/d[v0+2*v1] = C[g + 8*v1][2*t + v0]
//
// NOTE: this fixture validates the LIFTING PLUMBING (operand expansion, ABI,
// warp rendezvous). The fragment mapping itself appears on both sides of the
// comparison here; its ground truth is the CB_dgemm e2e test against the
// GH200 hardware reference.
extern "C" __global__ void kern_dmma16816(const double* A, const double* B,
                                          const double* C, double* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    double a[8], b[4], c[4], d[4];
#pragma unroll
    for (int v = 0; v < 8; ++v) a[v] = A[(g + 8 * (v & 1)) * 16 + (t + 4 * (v >> 1))];
#pragma unroll
    for (int v = 0; v < 4; ++v) b[v] = B[(t + 4 * v) * 8 + g];
#pragma unroll
    for (int v = 0; v < 4; ++v) c[v] = C[(g + 8 * (v >> 1)) * 8 + (2 * t + (v & 1))];

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7,%8,%9,%10,%11}, "
        "{%12,%13,%14,%15}, {%16,%17,%18,%19};\n"
        : "=d"(d[0]), "=d"(d[1]), "=d"(d[2]), "=d"(d[3])
        : "d"(a[0]), "d"(a[1]), "d"(a[2]), "d"(a[3]),
          "d"(a[4]), "d"(a[5]), "d"(a[6]), "d"(a[7]),
          "d"(b[0]), "d"(b[1]), "d"(b[2]), "d"(b[3]),
          "d"(c[0]), "d"(c[1]), "d"(c[2]), "d"(c[3]));

#pragma unroll
    for (int v = 0; v < 4; ++v) D[(g + 8 * (v >> 1)) * 8 + (2 * t + (v & 1))] = d[v];
}
