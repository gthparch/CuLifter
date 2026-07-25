// dual_hmma_frag — minimal: TWO HMMA.1688.F32.TF32 accumulating into ONE warp
// fragment, A operands loaded DIRECTLY from global (no smem, no LDSM). Isolates
// the dual-accumulate carry between two sequential HMMA into one C-fragment
// (the SYMM Mma1+Mma2 pattern) WITHOUT the LDSM/crosswise staging.
//   D = A1*B + A2*B + C ; both mma.m16n8k8 .row.col tf32.
// Frag: lane=4g+t; a0=A[g][t] a1=A[g+8][t] a2=A[g][t+4] a3=A[g+8][t+4];
//       b0=B[t][g] b1=B[t+4][g]; c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1].
#include <cstdint>
#include <cuda_runtime.h>
#define WARP 32
extern "C" __global__ void kern_dual_hmma_frag(
    const float* __restrict__ gA1, const float* __restrict__ gA2,
    const float* __restrict__ gB,  const float* __restrict__ gC,
    float* __restrict__ gD)
{
    const int lane = threadIdx.x & (WARP - 1);
    const int g = lane >> 2, t = lane & 3;
    uint32_t a0 = __float_as_uint(gA1[g*8 + t]);
    uint32_t a1 = __float_as_uint(gA1[(g+8)*8 + t]);
    uint32_t a2 = __float_as_uint(gA1[g*8 + t+4]);
    uint32_t a3 = __float_as_uint(gA1[(g+8)*8 + t+4]);
    uint32_t e0 = __float_as_uint(gA2[g*8 + t]);
    uint32_t e1 = __float_as_uint(gA2[(g+8)*8 + t]);
    uint32_t e2 = __float_as_uint(gA2[g*8 + t+4]);
    uint32_t e3 = __float_as_uint(gA2[(g+8)*8 + t+4]);
    uint32_t b0 = __float_as_uint(gB[t*8 + g]);
    uint32_t b1 = __float_as_uint(gB[(t+4)*8 + g]);
    float d0 = gC[g*8 + 2*t], d1 = gC[g*8 + 2*t+1];
    float d2 = gC[(g+8)*8 + 2*t], d3 = gC[(g+8)*8 + 2*t+1];
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d0),"f"(d1),"f"(d2),"f"(d3));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(e0),"r"(e1),"r"(e2),"r"(e3),"r"(b0),"r"(b1),
          "f"(d0),"f"(d1),"f"(d2),"f"(d3));
    gD[g*8 + 2*t]       = d0;
    gD[g*8 + 2*t+1]     = d1;
    gD[(g+8)*8 + 2*t]   = d2;
    gD[(g+8)*8 + 2*t+1] = d3;
}
