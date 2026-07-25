#include <cstdint>
extern "C" __global__
void ISA1b_atomicCAS_global(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Each thread tries to set out[0] to in[tid] if out[0] == 0
        atomicCAS(&out[0], 0, in[tid]);
    }
}
