#include <cstdint>
extern "C" __global__
void ISA4a_reduce_add(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        int sum = __reduce_add_sync(0xFFFFFFFF, val);
        if (threadIdx.x == 0) {
            out[0] = sum;
        }
    }
}
