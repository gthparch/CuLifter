// KC_nrm2_simple — partial-then-final L2 norm, mirrors CB_snrm2's
// launch-0/launch-1 split but with a minimal block-reduce we control.
//
// Shadows CB_snrm2 (which produces got=0 ref=32.7307 — the lifted
// nrm2_kernel doesn't populate workspace partials). If THIS kernel
// PASSes after lift, CB_snrm2's failure is in the cuBLAS-specific
// kernel template (heavy unroll + rsqrt-via-MUFU). If THIS FAILs,
// it's a real lifter bug in SMEM-reduce or workspace-store paths.
//
// Compute: result = sqrt(sum_i x[i]^2)
//
// Launch 0:  partial[blockIdx.x] = sum_{i in block stride} x[i]^2
//            grid = (N/BLOCK, 1, 1)  block = (BLOCK, 1, 1)
// Launch 1:  result = sqrt(sum_b partial[b])
//            grid = (1,1,1)  block = (BLOCK, 1, 1)

extern "C" {

struct KCNrm2PartialParams {
    const float* x;
    float* partial;     // out, 1 float per CTA
    int n;
};

struct KCNrm2FinalParams {
    const float* partial;
    float* result;      // out, 1 scalar
    int n_partials;
};

#define BLOCK 128

__global__ void kc_nrm2_partial(KCNrm2PartialParams p) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    // Strided accumulation across n
    float local = 0.0f;
    for (int i = gid; i < p.n; i += blockDim.x * gridDim.x) {
        float v = p.x[i];
        local += v * v;
    }

    // Plain SMEM tree reduce — no warp_shfl, no rsqrt, nothing exotic.
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) p.partial[blockIdx.x] = smem[0];
}

__global__ void kc_nrm2_final(KCNrm2FinalParams p) {
    __shared__ float smem[BLOCK];
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < p.n_partials; i += blockDim.x) local += p.partial[i];
    smem[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        // Use sqrtf — straightforward fp32 sqrt, NOT the rsqrt+mul
        // dance nvcc emits for cuBLAS nrm2_kernel.
        *p.result = __builtin_sqrtf(smem[0]);
    }
}

}  // extern "C"
