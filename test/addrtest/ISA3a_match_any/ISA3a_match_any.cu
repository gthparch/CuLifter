#include <cstdint>
extern "C" __global__
void ISA3a_match_any(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        int val = in[tid];
        unsigned mask = __match_any_sync(0xFFFFFFFF, val);
        out[tid] = __popc(mask);  // count how many threads have same value
    }
}
