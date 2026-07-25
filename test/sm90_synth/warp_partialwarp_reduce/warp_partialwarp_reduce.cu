// mb_partialwarp_then_full — minimal reproduction of the warp-reduction mass-drop
// under asymmetric (partial-warp) collectives followed by a barrier and a SECOND
// full-warp reduction.
//
// Trigger (mirrors so.193 softmax sum-stage): warp 0 lanes 0..7 do an EXTRA
// partial-warp block-reduce (advancing their collective count), then ALL lanes
// __syncthreads(), then a SECOND full-warp SHFL.DOWN reduce with NON-UNIFORM
// per-lane values. If the runtime models warp membership by a per-lane
// divergence/collective-count history, the post-barrier full-warp reduce in warp 0
// splits into {0..7} and {8..31} and drops mass. HW: barrier reconverges -> full warp.
//
// Block = 256 threads = 8 warps. Two block-reduces, each producing block-wide sum.
extern "C" __global__ void mb_partialwarp(const float* __restrict__ in,
                                          float* __restrict__ out, int N) {
    __shared__ float s[8];
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float a = in[tid];

    // ---- Block-reduce #1 (sum of `a`) ----
    // full-warp butterfly
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) a += __shfl_down_sync(0xffffffff, a, o);
    if (lane == 0) s[warp] = a;            // warp partial -> shared
    __syncthreads();
    // PARTIAL-WARP block reduce: ONLY warp0 lanes 0..7 read the 8 partials.
    float blocksum1 = 0.0f;
    if (warp == 0 && lane < 8) {
        float p = s[lane];
        // partial-warp (8-lane) reduce -> advances ONLY warp0 lanes 0..7's collective count
        p += __shfl_down_sync(0x000000ff, p, 4);
        p += __shfl_down_sync(0x000000ff, p, 2);
        p += __shfl_down_sync(0x000000ff, p, 1);
        if (lane == 0) s[0] = p;
    }
    __syncthreads();                       // RECONVERGE all 256 threads
    blocksum1 = s[0];

    // ---- Block-reduce #2 (sum of a DIFFERENT non-uniform quantity) ----
    // After the barrier, ALL lanes are converged in HW. The full-warp butterfly
    // below must see all 32 lanes of warp 0. NON-UNIFORM per-lane value `b`.
    float b = in[tid] * 1.5f - 0.25f * (float)lane;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) b += __shfl_down_sync(0xffffffff, b, o);
    if (lane == 0) s[warp] = b;
    __syncthreads();
    float blocksum2 = 0.0f;
    if (warp == 0 && lane < 8) {
        float p = s[lane];
        p += __shfl_down_sync(0x000000ff, p, 4);
        p += __shfl_down_sync(0x000000ff, p, 2);
        p += __shfl_down_sync(0x000000ff, p, 1);
        if (lane == 0) s[0] = p;
    }
    __syncthreads();
    blocksum2 = s[0];

    if (tid == 0) { out[0] = blocksum1; out[1] = blocksum2; }
}
