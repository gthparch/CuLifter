// KC_nrm2_with_imnmx — closer mimic of cuBLAS nrm2_kernel's
// thread-0-only IMNMX(bitcast(fabs(SMEM[0]))) chain.
//
// Mirrors the lifted IR pattern at BB_2020 of CB_snrm2:
//   - tid != 0 threads skip via predicated branch
//   - thread 0 reads SMEM[tid*4]=SMEM[0] (twice as lds.12 / lds.13)
//   - bitcast(fabs(.)) then IMNMX → integer min/max of the bits
//   - bitcast back to float, then predicated FMUL + STS + LDS + STG
//
// If KC_nrm2_with_imnmx FAILs producing CTA-invariant value → minimum
// reproducer for BUG-090. If it PASSes → bug is upstream (in the
// 7-stage SMEM reduce that produces SMEM[0]).

#include <stdint.h>

extern "C" {

struct KCNrm2ImnmxParams {
    const float* x;
    float* partial;
    int n;
};

#define BLOCK 128

__device__ __forceinline__ uint32_t bits_of(float v) {
    uint32_t u; __builtin_memcpy(&u, &v, 4); return u;
}
__device__ __forceinline__ float float_of(uint32_t u) {
    float f; __builtin_memcpy(&f, &u, 4); return f;
}

__global__ void kc_nrm2_with_imnmx(KCNrm2ImnmxParams p) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Strided sum-of-squares
    float local = 0.0f;
    for (int i = gid; i < p.n; i += blockDim.x * gridDim.x) {
        float v = p.x[i];
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();

    // 7-stage manual unrolled reduction
    if (tid < 64) smem[tid] += smem[tid + 64]; __syncthreads();
    if (tid < 32) smem[tid] += smem[tid + 32]; __syncthreads();
    if (tid < 16) smem[tid] += smem[tid + 16]; __syncthreads();
    if (tid < 8)  smem[tid] += smem[tid + 8];  __syncthreads();
    if (tid < 4)  smem[tid] += smem[tid + 4];  __syncthreads();
    if (tid < 2)  smem[tid] += smem[tid + 2];  __syncthreads();
    if (tid < 1)  smem[tid] += smem[tid + 1];  __syncthreads();

    // Mimic the cuBLAS BB_2020 pattern: ONLY thread 0 enters this branch.
    if (tid == 0) {
        // Both loads target SMEM[tid*4]=SMEM[0] when tid=0 — same as cuBLAS.
        float lds12 = smem[tid];   // SMEM[tid*4]
        float lds13 = smem[0];     // SMEM[0]   (FIXED address)

        // bitcast-fabs-IMNMX chain
        uint32_t bits_a = bits_of(__builtin_fabsf(lds12));
        uint32_t bits_b = bits_of(__builtin_fabsf(lds13));
        // IMNMX = signed-int compare → max
        int32_t cmp = (int32_t)bits_a < (int32_t)bits_b;
        uint32_t imnmx_max_12 = cmp ? bits_a : bits_b;
        uint32_t imnmx_max_13 = cmp ? bits_a : bits_b;
        float fmul_lhs    = float_of(imnmx_max_13);
        float fmul_rhs    = float_of(imnmx_max_12);   // R7 substitute
        float r10_stale   = float_of(imnmx_max_12);   // !P0 path

        // Predicate: (bitcast(imnmx_max.13) != 0)
        bool p0 = (float_of(imnmx_max_13) != 0.0f);
        float r10 = r10_stale;
        if (p0) {
            r10 = fmul_lhs * fmul_rhs;
        }

        smem[tid] = r10;            // unconditional STS [R3.X4]
        // No __syncthreads — same as cuBLAS pattern PC 0x2200..0x2210
        float r7 = smem[0];         // LDS R7, [RZ]
        p.partial[blockIdx.x] = r7;  // STG.E
    }
}

}  // extern "C"
