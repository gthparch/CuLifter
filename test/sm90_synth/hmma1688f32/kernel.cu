// HMMA.1688.F32 fixture (one warp): D(f32) = A(16x8 f16) * B(8x8 f16) + C(f32)
// mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 — fragment mapping per
// PTX ISA / cute MMA_Traits<SM80_16x8x8_F32F16F16F32_TN> (= the k<8 half of
// the 16816 layout):
//   lane = 4*g + t
//   a0={A[g][2t],A[g][2t+1]}  a1={A[g+8][2t],A[g+8][2t+1]}
//   b0={B[2t][g],B[2t+1][g]}
//   c0/d0=C[g][2t] c1/d1=C[g][2t+1] c2/d2=C[g+8][2t] c3/d3=C[g+8][2t+1]
#include <cuda_fp16.h>
extern "C" __global__ void kern_hmma1688f32(const __half* A, const __half* B,
                                            const float* C, float* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    unsigned a[2], b[1];
    float c[4], d[4];
    auto pack = [](__half lo, __half hi) {
        return (unsigned)__half_as_ushort(lo) |
               ((unsigned)__half_as_ushort(hi) << 16);
    };
    a[0] = pack(A[g * 8 + 2 * t],       A[g * 8 + 2 * t + 1]);
    a[1] = pack(A[(g + 8) * 8 + 2 * t], A[(g + 8) * 8 + 2 * t + 1]);
    b[0] = pack(B[(2 * t) * 8 + g],     B[(2 * t + 1) * 8 + g]);
    c[0] = C[g * 8 + 2 * t];
    c[1] = C[g * 8 + 2 * t + 1];
    c[2] = C[(g + 8) * 8 + 2 * t];
    c[3] = C[(g + 8) * 8 + 2 * t + 1];
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    D[g * 8 + 2 * t]           = d[0];
    D[g * 8 + 2 * t + 1]       = d[1];
    D[(g + 8) * 8 + 2 * t]     = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}
