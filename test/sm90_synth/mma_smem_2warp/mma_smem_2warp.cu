// mma_smem_2warp — MULTI-WARP cooperative cp.async (LDGSTS) staging into shared
// memory, then per-warp HMMA reading the shared tile.  Isolates the gap the
// 41_fmha audit hit: hmma_loop/hmma_ldsm pass at block=32 (single warp, NO
// LDGSTS); the real FMHA is 128 threads (4 warps) + heavy LDGSTS smem staging
// and produces warp-correlated inf on x86. This reproduces exactly that
// combination at minimal scale: 64 threads (2 warps), cp.async load of an A and
// a B tile from GLOBAL to shared, __syncthreads, then each warp does one
// m16n8k16 mma.sync over the shared tiles and writes its 4 accumulators.
//
// Output (per warp w in {0,1}): the 16x8 D = A_w * B_w GEMM, written as the
// standard mma.sync f32 fragment layout (4 floats/lane). out[w*128 + lane*4 + i].
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>

#define WARP 32
// Each warp gets its own A (16x16 f16) and B (16x8 f16) tile in GLOBAL memory.
// A_w: 16*16=256 halfs; B_w: 16*8=128 halfs. Two warps => g_a[2*256], g_b[2*128].
extern "C" __global__ void kern_mma_smem_2warp(
    const half* __restrict__ g_a,   // [2][16][16]
    const half* __restrict__ g_b,   // [2][16][8]
    float* __restrict__ d_out,      // [2][32][4]
    int N)
{
    // Shared staging: both warps' tiles, contiguous.
    // A region: 2*256 halfs = 512 halfs = 1024 B. B region: 2*128 = 256 halfs.
    __shared__ __align__(16) half sA[2*256];
    __shared__ __align__(16) half sB[2*128];

    int tid  = threadIdx.x;          // 0..63
    int warp = tid / WARP;           // 0 or 1
    int lane = tid % WARP;

    // COOPERATIVE cp.async: ALL 64 threads stage BOTH warps' tiles (cross-warp:
    // warp 0's threads help stage warp 1's tile and vice versa). 8 halfs (16B)
    // per async copy. Total A halfs = 512 -> 64 copies; each thread does 1.
    // (512 halfs / 8 per-copy = 64 copies = exactly one per thread.)
    {
        int idx = tid * 8;                     // half index into sA (0,8,...,504)
        __pipeline_memcpy_async(&sA[idx], &g_a[idx], 16);
    }
    // B halfs = 256 -> 32 copies; threads 0..31 do them (warp 0 stages ALL B).
    if (tid < 32) {
        int idx = tid * 8;
        __pipeline_memcpy_async(&sB[idx], &g_b[idx], 16);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    // Per-warp MMA over the SHARED tiles. Fragment loads via ldmatrix from the
    // warp's own region of shared memory.
    half* aw = &sA[warp * 256];      // this warp's A tile (16x16)
    half* bw = &sB[warp * 128];      // this warp's B tile (16x8)

    // ldmatrix.x4 A: m16n16 needs 4 fragments; each lane row = lane%16, the
    // x4 form loads 4 8x8 sub-tiles. Row stride 16 halfs = 32 B.
    uint32_t a_addr = (uint32_t)__cvta_generic_to_shared(&aw[(lane % 16) * 16 + (lane / 16) * 8]);
    uint32_t a0,a1,a2,a3;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(a_addr));
    // ldmatrix.x2 B: 16x8 -> 2 fragments. Row stride 8 halfs = 16 B.
    uint32_t b_addr = (uint32_t)__cvta_generic_to_shared(&bw[(lane % 16) * 8]);
    uint32_t b0,b1;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(b0),"=r"(b1) : "r"(b_addr));

    float d0=0,d1=0,d2=0,d3=0;
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(0.f),"f"(0.f),"f"(0.f),"f"(0.f));

    float* o = &d_out[warp * 128 + lane * 4];
    o[0]=d0; o[1]=d1; o[2]=d2; o[3]=d3;
}
