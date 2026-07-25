// hmma_ldsm_tf32_val — VALUE-verifying single HMMA.1688.F32.TF32 fed by
// LDSM.16.M88.4 with NON-UNIFORM data (the missing coverage: hmma_ldsm uses
// uniform A=1/B=2 + only an isnan check, so it can't catch a fragment
// permutation). D = A(16x8 tf32)*B(8x8) + C, A staged in smem so that the
// STANDARD ldmatrix.x4 yields a0=A[g][t] a1=A[g+8][t] a2=A[g][t+4] a3=A[g+8][t+4].
//
// Staging recipe (derived from ldmatrix M88 semantics + the SM80_16x8x8 A frag):
//   reg m of lane 4g+t = word t of the 16 bytes at lane (8m+g)'s addr.
//   So lane L=8m+g stages 4 tf32 = A[ (m&1?g+8:g) ][ (m>=2?4:0) + 0..3 ].
#include <cstdint>
#include <cuda_runtime.h>
#define WARP 32
static __device__ __forceinline__ uint32_t cvta_s(const void* p){return (uint32_t)__cvta_generic_to_shared(p);}
extern "C" __global__ void kern_hmma_ldsm_tf32_val(
    const float* __restrict__ gA, const float* __restrict__ gB,
    const float* __restrict__ gC, float* __restrict__ gD)
{
    __shared__ __align__(16) float sA[32*4];  // 32 lanes x 4 tf32 = the x4 staging
    const int lane = threadIdx.x & (WARP-1);
    const int g = lane >> 2, t = lane & 3;
    // Stage A: lane L=8m+g stages its 16-byte row.
    {
        int m = lane >> 3;        // 0..3
        int gg = lane & 7;        // 0..7 (row within an 8-row matrix)
        int rowA = (m & 1) ? gg + 8 : gg;
        int colB = (m >= 2) ? 4 : 0;
        sA[lane*4 + 0] = gA[rowA*8 + colB + 0];
        sA[lane*4 + 1] = gA[rowA*8 + colB + 1];
        sA[lane*4 + 2] = gA[rowA*8 + colB + 2];
        sA[lane*4 + 3] = gA[rowA*8 + colB + 3];
    }
    __syncwarp();
    uint32_t a0,a1,a2,a3;
    uint32_t aa = cvta_s(&sA[lane*4]);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a0),"=r"(a1),"=r"(a2),"=r"(a3) : "r"(aa));
    uint32_t b0 = __float_as_uint(gB[t*8 + g]);
    uint32_t b1 = __float_as_uint(gB[(t+4)*8 + g]);
    float d0=gC[g*8+2*t], d1=gC[g*8+2*t+1], d2=gC[(g+8)*8+2*t], d3=gC[(g+8)*8+2*t+1];
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d0),"=f"(d1),"=f"(d2),"=f"(d3)
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
          "f"(d0),"f"(d1),"f"(d2),"f"(d3));
    gD[g*8+2*t]=d0; gD[g*8+2*t+1]=d1; gD[(g+8)*8+2*t]=d2; gD[(g+8)*8+2*t+1]=d3;
}
