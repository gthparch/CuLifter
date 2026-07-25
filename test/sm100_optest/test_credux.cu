#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

extern "C" __global__ void test_credux(int* out, const int* in, int n) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int val = (i < n) ? in[i] : 0;
    // cg::reduce should generate CREDUX on SM100
    int sum = cg::reduce(tile, val, cg::plus<int>());
    if (tile.thread_rank() == 0) atomicAdd(out, sum);
}
