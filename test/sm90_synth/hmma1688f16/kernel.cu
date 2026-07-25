// HMMA.1688.F16 fixture (one warp): D(f16) = A(16x8 f16) * B(8x8 f16) + C(f16)
// mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 — fragment mapping per
// PTX ISA / cute MMA_Traits<SM80_16x8x8_F16F16F16F16_TN>:
//   lane = 4*g + t
//   a0={A[g][2t],A[g][2t+1]}  a1={A[g+8][2t],A[g+8][2t+1]}
//   b0={B[2t][g],B[2t+1][g]}
//   c0/d0={C[g][2t],C[g][2t+1]}  c1/d1={C[g+8][2t],C[g+8][2t+1]}  (packed v2f16)
#include <cuda_fp16.h>
extern "C" __global__ void kern_hmma1688f16(const __half* A, const __half* B,
                                            const __half* C, __half* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    unsigned a[2], b[1], c[2], d[2];
    auto pack = [](__half lo, __half hi) {
        return (unsigned)__half_as_ushort(lo) |
               ((unsigned)__half_as_ushort(hi) << 16);
    };
    a[0] = pack(A[g * 8 + 2 * t],       A[g * 8 + 2 * t + 1]);
    a[1] = pack(A[(g + 8) * 8 + 2 * t], A[(g + 8) * 8 + 2 * t + 1]);
    b[0] = pack(B[(2 * t) * 8 + g],     B[(2 * t + 1) * 8 + g]);
    c[0] = pack(C[g * 8 + 2 * t],       C[g * 8 + 2 * t + 1]);
    c[1] = pack(C[(g + 8) * 8 + 2 * t], C[(g + 8) * 8 + 2 * t + 1]);
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3}, {%4}, {%5,%6};\n"
        : "=r"(d[0]), "=r"(d[1])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
    D[g * 8 + 2 * t]           = __ushort_as_half((unsigned short)(d[0] & 0xffff));
    D[g * 8 + 2 * t + 1]       = __ushort_as_half((unsigned short)(d[0] >> 16));
    D[(g + 8) * 8 + 2 * t]     = __ushort_as_half((unsigned short)(d[1] & 0xffff));
    D[(g + 8) * 8 + 2 * t + 1] = __ushort_as_half((unsigned short)(d[1] >> 16));
}
