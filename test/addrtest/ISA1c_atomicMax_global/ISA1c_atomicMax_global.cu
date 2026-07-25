#include <cstdint>
extern "C" __global__
void ISA1c_atomicMax_global(int* out, const int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        atomicMax(&out[0], in[tid]);
    }
}
