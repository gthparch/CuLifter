// mma_chain_frag — CHAINED 2-MMA microbench isolating the 41_fmha forward
// chain link: MMA1 output fragment (P = X*Yt) --(in-fragment nonlinear)-->
// F2FP f32->f16 --> STS to smem --> LDSM.16.MT88 (TRANSPOSED reload) -->
// MMA2 B-operand --> O = V * P  (both mma.sync.m16n8k8, HMMA.1688.F32).
//
// This is the EXACT chain the 41_fmha forward kernel uses (salvage_k1):
//   MMA1: HMMA.1688.F32 accumulating P into 4 accum/lane (f32)
//   softmax-role: in-fragment scale+ReLU makes P NON-UNIFORM + data-dependent
//   F2FP.F16.F32.PACK_AB: P f32 -> packed f16
//   STS: store packed-f16 P to smem
//   __syncthreads
//   MMA2: LDSM.16.MT88.4 reloads P (TRANSPOSED) as the HMMA B-operand,
//         V loaded plainly as the A-operand; O accumulates into 4 accum/lane.
//
// Single warp (32 threads). NON-UNIFORM distinct inputs. The host driver
// (host_cpu.cpp) replays the WHOLE chain in f64 and compares O PER ROW. The
// goal is to decide whether the lifter's chained fragment->register->fragment
// path is correct under non-uniform data (single-MMA fixtures already are).
//
// Fragment layouts (PTX mma.m16n8k8 .row.col, lane = 4*g + t, g=lane>>2, t=lane&3):
//   A (16x8): a0={A[g][2t],A[g][2t+1]}  a1={A[g+8][2t],A[g+8][2t+1]}
//   B ( 8x8): b0={B[2t][g],B[2t+1][g]}
//   D (16x8): d0={D[g][2t],D[g][2t+1]}  d1={D[g+8][2t],D[g+8][2t+1]}  (f32)
//
// MMA1 produces P(16x8). We store P row-major to smem (sP[16][8] halfs). MMA2
// needs P as its 8x8 B-operand TRANSPOSED-reloaded; we form the 8x8 V*P tile
// MMA2 = V(16x8) * P_used(8x8). To exercise MT88 we reload the B-operand of
// MMA2 via ldmatrix.trans from the row-major smem P tile, exactly like 41.

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define WARP 32

extern "C" __global__ void kern_mma_chain_frag(
    const half*  __restrict__ g_X,   // [16][8] f16   (MMA1 A)
    const half*  __restrict__ g_Y,   // [8][8]  f16   (MMA1 B)  -- P = X * Y
    const half*  __restrict__ g_V,   // [16][8] f16   (MMA2 A)
    float*       __restrict__ g_O,   // [16][8] f32   (MMA2 D = V * Pt)  fragment-order out
    float*       __restrict__ g_P,   // [16][8] f32   debug: MMA1 P (post-nonlinear) fragment-order
    int N)
{
    __shared__ __align__(16) half sP[16 * 8];   // P staged row-major (post f2fp)

    const int lane = threadIdx.x & (WARP - 1);
    const int g = lane >> 2;     // 0..7
    const int t = lane & 3;      // 0..3

    // ---- Load MMA1 A (X 16x8) fragments: a0 rows g, a1 rows g+8 ----
    auto packh = [](half lo, half hi) {
        return (uint32_t)__half_as_ushort(lo) | ((uint32_t)__half_as_ushort(hi) << 16);
    };
    uint32_t xa0 = packh(g_X[g * 8 + 2 * t],       g_X[g * 8 + 2 * t + 1]);
    uint32_t xa1 = packh(g_X[(g + 8) * 8 + 2 * t], g_X[(g + 8) * 8 + 2 * t + 1]);
    // ---- Load MMA1 B (Y 8x8) fragment: b0 = {Y[2t][g], Y[2t+1][g]} ----
    uint32_t yb0 = packh(g_Y[(2 * t) * 8 + g], g_Y[(2 * t + 1) * 8 + g]);

    // ---- MMA1: P(16x8) = X * Y ----
    float p0 = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(p0), "=f"(p1), "=f"(p2), "=f"(p3)
        : "r"(xa0), "r"(xa1), "r"(yb0),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));

    // ---- In-fragment NON-LINEAR (mirrors softmax making P non-uniform/data-dep) ----
    // scale + ReLU per accumulator element. Keeps an exact oracle while making
    // the values genuinely non-uniform and data-dependent (so any frag->row
    // permutation in the chain is exposed).
    auto nl = [](float x) { float y = 0.5f * x + 0.25f; return y > 0.f ? y : 0.f; };
    p0 = nl(p0); p1 = nl(p1); p2 = nl(p2); p3 = nl(p3);

    // Debug emit P (fragment order) for diagnosis. d0 rows g, d1 rows g+8.
    if (g_P) {
        g_P[(g) * 8 + 2 * t]         = p0;
        g_P[(g) * 8 + 2 * t + 1]     = p1;
        g_P[(g + 8) * 8 + 2 * t]     = p2;
        g_P[(g + 8) * 8 + 2 * t + 1] = p3;
    }

    // ---- F2FP P f32 -> f16, then STS to smem (row-major P[16][8]) ----
    // Each lane writes its 4 P elements to their (row,col) slots in sP. This is
    // the lifter's F2FP (per-lane pack) + STS (real scalar stores) path.
    half hp0 = __float2half(p0), hp1 = __float2half(p1);
    half hp2 = __float2half(p2), hp3 = __float2half(p3);
    sP[(g) * 8 + 2 * t]         = hp0;
    sP[(g) * 8 + 2 * t + 1]     = hp1;
    sP[(g + 8) * 8 + 2 * t]     = hp2;
    sP[(g + 8) * 8 + 2 * t + 1] = hp3;

    __syncthreads();

    // ---- Load MMA2 A (V 16x8) fragments plainly: a0 rows g, a1 rows g+8 ----
    uint32_t va0 = packh(g_V[g * 8 + 2 * t],       g_V[g * 8 + 2 * t + 1]);
    uint32_t va1 = packh(g_V[(g + 8) * 8 + 2 * t], g_V[(g + 8) * 8 + 2 * t + 1]);

    // ---- MMA2 B-operand: reload P from smem via LDSM.16.MT88 (TRANSPOSED) ----
    // ldmatrix.trans m8n8: lane L addresses ROW L of the 8x8 tile it loads. For
    // an 8x8 tile we use lanes 0..7's addresses (the .x1 form). The transposed
    // load redistributes so that the B-fragment b0 of lane L = {Pt[2t][g], Pt[2t+1][g]}
    // = {P[g][2t], P[g][2t+1]} ... i.e. it reads the row-major P tile as Pt for
    // the B-operand. We point each lane at sP row (lane & 7) (8 rows), stride 8 halfs.
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(&sP[(lane & 7) * 8]);
    uint32_t pb0;
    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];"
                 : "=r"(pb0) : "r"(addr));

    // ---- MMA2: O(16x8) = V * Pt ----
    float o0 = 0.f, o1 = 0.f, o2 = 0.f, o3 = 0.f;
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=f"(o0), "=f"(o1), "=f"(o2), "=f"(o3)
        : "r"(va0), "r"(va1), "r"(pb0),
          "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));

    // ---- Emit O in fragment order (d0 rows g, d1 rows g+8) ----
    g_O[(g) * 8 + 2 * t]         = o0;
    g_O[(g) * 8 + 2 * t + 1]     = o1;
    g_O[(g + 8) * 8 + 2 * t]     = o2;
    g_O[(g + 8) * 8 + 2 * t + 1] = o3;
}
