#include <cstdint>
extern "C" __global__
void ISA4b_reduce_min_max(int* out_min, int* out_max, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        int mn = __reduce_min_sync(0xFFFFFFFF, val);
        int mx = __reduce_max_sync(0xFFFFFFFF, val);
        if (threadIdx.x == 0) {
            out_min[0] = mn;
            out_max[0] = mx;
        }
    }
}
