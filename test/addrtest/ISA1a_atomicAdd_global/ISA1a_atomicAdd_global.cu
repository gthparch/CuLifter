#include <cstdint>
extern "C" __global__
void ISA1a_atomicAdd_global(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicAdd(&out[0], in[tid]);
    }
}
