__global__ void warp_prefix_sum(const int* __restrict__ in,
                                int* __restrict__ out,
                                int n)
{
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int lane   = threadIdx.x & 31;   // lane id in warp
    unsigned m = __activemask();

    int x = (tid < n) ? in[tid] : 0;

    // inclusive scan within each warp
    // after this, x = sum of in[warp_base..tid] within the warp
    for (int offset = 1; offset < 32; offset <<= 1) {
        int y = __shfl_up_sync(m, x, offset);
        if (lane >= offset)
            x += y;
    }

    if (tid < n)
        out[tid] = x;
}
