// KC_nrm2_unrolled — closer mimic of cuBLAS nrm2_kernel's SASS pattern.
//
// Adds vs KC_nrm2_simple (which PASSes):
//   - Manual 7-stage unrolled SMEM tree reduce (mimics cuBLAS heavy unroll
//     that produces 8 sts_shared_addr_cast stores in lifted IR).
//   - Predicated final FMUL followed by UNCONDITIONAL store (mimics the
//     @P0 FMUL R10 / STS [R3.X4], R10 / LDS R7,[RZ] / STG pattern at PC
//     0x21b0..0x2230 in a cuBLAS nrm2 kernel).
//   - rsqrt-based scaling (the SASS suspect uses R12 to gate; we use the
//     accumulator value to mirror that).
//
// If THIS kernel PASSes after lift+run, the cuBLAS bug is in some
// other cuBLAS-internal pattern (instrumentation needed). If THIS
// FAILs with a similar got≪ref pattern, we have a 50-line minimum
// reproducer for BUG-090.

extern "C" {

struct KCNrm2UnrolledParams {
    const float* x;
    float* partial;     // out, 1 float per CTA
    int n;
};

#define BLOCK 128

__global__ void kc_nrm2_unrolled(KCNrm2UnrolledParams p) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Strided sum of squares
    float local = 0.0f;
    for (int i = gid; i < p.n; i += blockDim.x * gridDim.x) {
        float v = p.x[i];
        local += v * v;
    }
    smem[tid] = local;
    __syncthreads();

    // 7-stage manual unrolled reduction.
    if (tid < 64) smem[tid] += smem[tid + 64]; __syncthreads();
    if (tid < 32) smem[tid] += smem[tid + 32]; __syncthreads();
    if (tid < 16) smem[tid] += smem[tid + 16]; __syncthreads();
    if (tid < 8)  smem[tid] += smem[tid + 8];  __syncthreads();
    if (tid < 4)  smem[tid] += smem[tid + 4];  __syncthreads();
    if (tid < 2)  smem[tid] += smem[tid + 2];  __syncthreads();
    if (tid < 1)  smem[tid] += smem[tid + 1];  __syncthreads();

    // Mimic the cuBLAS @P0 FMUL / unconditional STS / LDS / STG pattern.
    // smem[0] now holds the per-block sum.
    //
    //  @P0 FMUL R10, R2, R7        (predicated)
    //  STS [R3.X4], R10            (UNCONDITIONAL — uses stale R10 if !P0)
    //  LDS R7, [RZ]                (read smem[0])
    //  STG.E [R4.64], R7           (write workspace[blockIdx])
    //
    // Here we choose: predicate = (smem[0] != 0). For thread 0 the
    // predicate gates a multiply by 1.0; for other threads the
    // predicate is also true (smem[0] is the same per-block scalar).
    // The interesting case for the lifter is the dataflow: r10
    // depends on `local` (per-thread stale) when predicate is false.
    float r10 = local;             // stale per-thread value
    bool p0 = (smem[0] != 0.0f);
    if (p0) r10 = smem[0] * 1.0f;  // @P0 FMUL pattern
    smem[tid] = r10;               // unconditional STS [R3.X4], R10

    // No __syncthreads here — same as cuBLAS PC 0x2200..0x2210
    // (relies on each thread reading what THIS thread wrote to its
    //  own SMEM slot earlier in the unrolled chain).
    float r7 = smem[0];            // LDS R7, [RZ]
    p.partial[blockIdx.x] = r7;    // STG.E [R4.64], R7

    // (Real cuBLAS nrm2_kernel has launch-1 do the sqrt; we just
    //  test the partial-write pattern here.)
}

}  // extern "C"
