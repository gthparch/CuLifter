#include <cstdint>
extern "C" __global__
void ISA5a_brev_clz_popc(int* out_brev, int* out_clz, int* out_popc,
                          const unsigned int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int val = in[tid];
        out_brev[tid] = (int)__brev(val);
        out_clz[tid]  = __clz(val);
        out_popc[tid] = __popc(val);
    }
}
