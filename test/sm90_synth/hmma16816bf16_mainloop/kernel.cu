// BF16 GEMM MAINLOOP audit: m16n8k(16*KT) bf16-input / f32-accum.
// One warp accumulates KT distinct k-tiles of HMMA.16816.F32.BF16 into a
// single f32 fragment (the cutlass MmaWithReductionMultistage inner loop shape:
// repeated HMMA.16816.F32.BF16 with the SAME f32 accumulator fed back, DISTINCT
// A/B bf16 fragments per k-tile). The accumulator feedback + bf16-input decode
// (bits<<16 on each lane) + multi-tile sum are the distinct surface. C := 0;
// each iter does D = A_i (16x16 bf16) * B_i (16x8 bf16) + D.
//
// Fragment->thread mapping is the standard PTX m16n8k16 layout (same as the
// hmma16816bf16 atom fixture). A_i/B_i for tile i are read at offset
// i*(per-warp-fragment-size) so every tile contributes a DIFFERENT product.
#include <cuda_bf16.h>
extern "C" __global__ void kern_bf16_mainloop(
    const __nv_bfloat16* __restrict__ A,   // [KT][16][16] row-major per tile
    const __nv_bfloat16* __restrict__ B,   // [KT][16][8]  (k-major: B[k][n])
    float* __restrict__ D,                  // [16][8]
    int KT)
{
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    auto pack = [](__nv_bfloat16 lo, __nv_bfloat16 hi) {
        return (unsigned)__bfloat16_as_ushort(lo) |
               ((unsigned)__bfloat16_as_ushort(hi) << 16);
    };
    float d0=0.f, d1=0.f, d2=0.f, d3=0.f;
    for (int i = 0; i < KT; ++i) {
        const __nv_bfloat16* At = A + i*16*16;
        const __nv_bfloat16* Bt = B + i*16*8;
        unsigned a0 = pack(At[g*16 + 2*t],       At[g*16 + 2*t + 1]);
        unsigned a1 = pack(At[(g+8)*16 + 2*t],   At[(g+8)*16 + 2*t + 1]);
        unsigned a2 = pack(At[g*16 + 2*t + 8],   At[g*16 + 2*t + 9]);
        unsigned a3 = pack(At[(g+8)*16 + 2*t + 8], At[(g+8)*16 + 2*t + 9]);
        unsigned b0 = pack(Bt[(2*t)*8 + g],     Bt[(2*t + 1)*8 + g]);
        unsigned b1 = pack(Bt[(2*t + 8)*8 + g], Bt[(2*t + 9)*8 + g]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    D[g*8 + 2*t]       = d0;
    D[g*8 + 2*t + 1]   = d1;
    D[(g+8)*8 + 2*t]   = d2;
    D[(g+8)*8 + 2*t + 1] = d3;
}
