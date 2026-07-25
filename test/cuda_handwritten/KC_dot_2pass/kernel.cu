// KC_dot_2pass — 2-launch dot product: per-block partial sums via warp_shfl
// reduction, then second kernel reduces partials to a single scalar.
//
// Shadows CB_sdot's 2-kernel chain but with a simple, hand-written
// reduction so we can isolate lifter handling of warp_shfl.

extern "C" {

struct KCDotPartialParams {
    const float* x;
    const float* y;
    float* partial;     // out: 1 float per CTA
    int n;
};

struct KCDotReduceParams {
    const float* partial;
    float* result;      // out: 1 scalar
    int n_partials;
};

// Launch 0: each CTA computes partial += sum_{i in cta range} x[i] * y[i]
__global__ void kc_dot_partial(KCDotPartialParams p) {
    int tid    = threadIdx.x;
    int gid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = gid; i < p.n; i += stride) local += p.x[i] * p.y[i];

    // Warp-level reduction (single warp per block for simplicity)
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);

    if (tid == 0) p.partial[blockIdx.x] = local;
}

// Launch 1: single-block reduction over the partial array
__global__ void kc_dot_reduce(KCDotReduceParams p) {
    int tid = threadIdx.x;
    float local = 0.0f;
    for (int i = tid; i < p.n_partials; i += blockDim.x) local += p.partial[i];
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);
    if (tid == 0) *p.result = local;
}

}  // extern "C"
