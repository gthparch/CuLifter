// HMMA.1688.F32.TF32 fixture (one warp): D(f32) = A(16x8 tf32) * B(8x8 tf32) + C(f32)
// mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 — fragment mapping per
// PTX ISA / cute MMA_Traits<SM80_16x8x8_F32TF32TF32F32_TN>
// (ARegisters=uint32_t[4], BRegisters=uint32_t[2]; one tf32 per full 32-bit reg):
//   lane = 4*g + t
//   a0=A[g][t]  a1=A[g+8][t]  a2=A[g][t+4]  a3=A[g+8][t+4]
//   b0=B[t][g]  b1=B[t+4][g]
//   c0/d0=C[g][2t] c1/d1=C[g][2t+1] c2/d2=C[g+8][2t] c3/d3=C[g+8][2t+1]
// Host pre-snaps A/B to the tf32 grid (low 13 mantissa bits zero), matching
// the cvt.rna.tf32-converted operand domain of real CUTLASS/cuBLAS kernels.
extern "C" __global__ void kern_hmma1688tf32(const float* A, const float* B,
                                             const float* C, float* D)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    unsigned a[4], b[2];
    float c[4], d[4];
    a[0] = __float_as_uint(A[g * 8 + t]);
    a[1] = __float_as_uint(A[(g + 8) * 8 + t]);
    a[2] = __float_as_uint(A[g * 8 + t + 4]);
    a[3] = __float_as_uint(A[(g + 8) * 8 + t + 4]);
    b[0] = __float_as_uint(B[t * 8 + g]);
    b[1] = __float_as_uint(B[(t + 4) * 8 + g]);
    c[0] = C[g * 8 + 2 * t];
    c[1] = C[g * 8 + 2 * t + 1];
    c[2] = C[(g + 8) * 8 + 2 * t];
    c[3] = C[(g + 8) * 8 + 2 * t + 1];
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
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
