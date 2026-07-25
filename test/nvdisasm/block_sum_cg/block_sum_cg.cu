#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void block_sum_cg(const float* __restrict__ in,
                             float* __restrict__ block_sums,
                             int n)
{
    cg::thread_block cta = cg::this_thread_block();

    extern __shared__ float sdata[];

    int tid   = blockIdx.x * blockDim.x + threadIdx.x;
    int lane  = threadIdx.x;

    float val = (tid < n) ? in[tid] : 0.0f;
    sdata[lane] = val;
    cg::sync(cta);

    // simple shared-memory reduction within the block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride)
            sdata[lane] += sdata[lane + stride];
        cg::sync(cta);
    }

    if (lane == 0)
        block_sums[blockIdx.x] = sdata[0];
}
