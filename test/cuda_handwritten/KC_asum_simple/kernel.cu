// KC_asum_simple — single-block absolute-sum (|x[0]| + |x[1]| + ... + |x[n-1]|).
// Tiny enough to verify with a single thread block. Exercises warp shuffles +
// shared mem (block-wide reduction).

extern "C" {

struct KCAsumParams {
    const float* x;
    float* result;
    int n;
};

__global__ void kc_asum_simple(KCAsumParams p) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;

    // Strided accumulation across n by single block of blockDim threads
    float local = 0.0f;
    for (int i = tid; i < p.n; i += blockDim.x) {
        float v = p.x[i];
        local += (v < 0.0f) ? -v : v;
    }

    // Warp reduce
    for (int off = 16; off > 0; off >>= 1)
        local += __shfl_down_sync(0xffffffff, local, off);

    int warp_id = tid >> 5;
    int lane    = tid & 31;
    if (lane == 0) smem[warp_id] = local;
    __syncthreads();

    // Final reduce: warp 0 sums the per-warp partials
    if (warp_id == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        local = (lane < n_warps) ? smem[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1)
            local += __shfl_down_sync(0xffffffff, local, off);
        if (lane == 0) *p.result = local;
    }
}

}  // extern "C"
