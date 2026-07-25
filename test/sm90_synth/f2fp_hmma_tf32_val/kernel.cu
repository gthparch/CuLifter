// f2fp_hmma_tf32_val — VALUE-verifying F2FP.TF32 -> HMMA.1688.F32.TF32 COMPOSITION.
//
// The coverage gap this closes: f2fp_tf32_packb tests the F2FP.TF32 convert in
// ISOLATION (convert + store); hmma_ldsm_tf32_val tests the HMMA atom fed by
// ALREADY-tf32 bit patterns. NEITHER tests the COMPOSITION the real tf32 GEMMs
// (cutlass 14_ampere_tf32, 27 Gemm_1xTF32, cuBLAS s1688gemm_tf32) actually run:
//   f32 operand -> cvt.rn.tf32.f32 (F2FP.TF32.F32.PACK_B, HW RN-even) -> the
//   m16n8k8 f32.tf32.tf32 tensor op -> f32 accumulate.
//
// This is the exact interaction the F2FP.TF32 RN-even fix (lifter.py lift_f2f TF32
// branch: u + 0x0FFF + ((u>>13)&1) ; & 0xFFFFE000) must get right TOGETHER WITH the
// x86 hmma1688_tf32_warp wrapper's &0xFFFFE000 mask (which is an identity on a valid
// RN-even tf32 limb). A single warp computes one m16n8k8 tf32 MMA:
//   D[16x8] = A[16x8 tf32] * B[8x8 tf32] + C[16x8]   (f32 accumulate)
// where A,B come in as RAW f32 and are F2FP.TF32-converted in-kernel.
//
// The host oracle (host_cpu.cpp) feeds inputs whose mantissa bit12 (0x1000) is set,
// so RN-even and plain TRUNCATION DIFFER by 1 tf32-ULP -> a truncating lift FAILS,
// only the RN-even fix passes. SM80_16x8x8_F32TF32TF32F32_TN fragment map:
//   A: a0=A[g][t] a1=A[g+8][t] a2=A[g][t+4] a3=A[g+8][t+4]   (g=lane>>2, t=lane&3)
//   B: b0=B[t][g] b1=B[t+4][g]
//   C/D: d0=C[g][2t] d1=C[g][2t+1] d2=C[g+8][2t] d3=C[g+8][2t+1]
#include <cstdint>
#include <cuda_runtime.h>
#define WARP 32

// f32 -> tf32 via the corpus-fused HW op (cvt.rn.tf32.f32 -> F2FP.TF32.F32.PACK_B).
static __device__ __forceinline__ uint32_t to_tf32(float v) {
    uint32_t p;
    asm volatile("cvt.rn.tf32.f32 %0, %1;" : "=r"(p) : "f"(v));
    return p;
}

extern "C" __global__ void kern_f2fp_hmma_tf32_val(
    const float* __restrict__ gA,   // 16x8 row-major f32
    const float* __restrict__ gB,   // 8x8 row-major f32
    const float* __restrict__ gC,   // 16x8 row-major f32
    float* __restrict__ gD)         // 16x8 row-major f32 out
{
    const int lane = threadIdx.x & (WARP - 1);
    const int g = lane >> 2, t = lane & 3;
    // Convert each operand limb f32 -> tf32 with the HW F2FP.TF32 (RN-even).
    uint32_t a0 = to_tf32(gA[(g)*8 + t]);
    uint32_t a1 = to_tf32(gA[(g + 8)*8 + t]);
    uint32_t a2 = to_tf32(gA[(g)*8 + (t + 4)]);
    uint32_t a3 = to_tf32(gA[(g + 8)*8 + (t + 4)]);
    uint32_t b0 = to_tf32(gB[t*8 + g]);
    uint32_t b1 = to_tf32(gB[(t + 4)*8 + g]);
    float d0 = gC[g*8 + 2*t],   d1 = gC[g*8 + 2*t + 1];
    float d2 = gC[(g + 8)*8 + 2*t], d3 = gC[(g + 8)*8 + 2*t + 1];
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
          "f"(d0), "f"(d1), "f"(d2), "f"(d3));
    gD[g*8 + 2*t] = d0;       gD[g*8 + 2*t + 1] = d1;
    gD[(g + 8)*8 + 2*t] = d2; gD[(g + 8)*8 + 2*t + 1] = d3;
}
