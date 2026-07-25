__global__ void warp_any_nonzero(const int* __restrict__ flags,
                                 int* __restrict__ warp_any,
                                 int n)
{
    int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    int lane    = threadIdx.x & 31;
    int warpId  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    unsigned m  = __activemask();

    int pred = (tid < n) ? (flags[tid] != 0) : 0;

    unsigned mask = __ballot_sync(m, pred); // warp vote

    if (lane == 0)
        warp_any[warpId] = (mask != 0);
}
