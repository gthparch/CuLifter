__global__ void warp_reduce_sum(const float* __restrict__ in,
                                float* __restrict__ warp_sums,
                                int n)
{
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int lane    = threadIdx.x & 31;          // lane id in warp
    int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned m  = __activemask();

    float val = (tid < n) ? in[tid] : 0.0f;

    // warp-wide tree reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(m, val, offset);

    if (lane == 0)
        warp_sums[warpId] = val;
}
