// HMMA.16816.F32.BF16 fixture (one warp): D(f32) = A(16x16 bf16) * B(16x8 bf16) + C(f32)
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 — fragment mapping per PTX
// (identical A/B layout to the f16 form; C/D are f32 scalars):
//   lane = 4*g + t
//   a0={A[g][2t],A[g][2t+1]} a1={A[g+8][2t],...} a2/a3 = same +8 cols
//   b0={B[2t][g],B[2t+1][g]} b1={B[2t+8][g],B[2t+9][g]}
//   c0/d0=C[g][2t] c1/d1=C[g][2t+1] c2/d2=C[g+8][2t] c3/d3=C[g+8][2t+1]
#include <cuda_bf16.h>
extern "C" __global__ void kern_hmma16816bf16(const __nv_bfloat16* A,
                                              const __nv_bfloat16* B,
                                              const float* C, float* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    unsigned a[4], b[2];
    float c[4], d[4];
    auto pack = [](__nv_bfloat16 lo, __nv_bfloat16 hi) {
        return (unsigned)__bfloat16_as_ushort(lo) |
               ((unsigned)__bfloat16_as_ushort(hi) << 16);
    };
    a[0] = pack(A[g * 16 + 2 * t],       A[g * 16 + 2 * t + 1]);
    a[1] = pack(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]);
    a[2] = pack(A[g * 16 + 2 * t + 8],   A[g * 16 + 2 * t + 9]);
    a[3] = pack(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]);
    b[0] = pack(B[(2 * t) * 8 + g],     B[(2 * t + 1) * 8 + g]);
    b[1] = pack(B[(2 * t + 8) * 8 + g], B[(2 * t + 9) * 8 + g]);
    c[0] = C[g * 8 + 2 * t];
    c[1] = C[g * 8 + 2 * t + 1];
    c[2] = C[(g + 8) * 8 + 2 * t];
    c[3] = C[(g + 8) * 8 + 2 * t + 1];
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
    D[g * 8 + 2 * t]           = d[0];
    D[g * 8 + 2 * t + 1]       = d[1];
    D[(g + 8) * 8 + 2 * t]     = d[2];
    D[(g + 8) * 8 + 2 * t + 1] = d[3];
}
