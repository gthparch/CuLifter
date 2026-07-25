#include <cstdint>

__device__ __forceinline__ unsigned int device_bfe(unsigned int val, unsigned int pos, unsigned int len) {
    unsigned int result;
    asm("bfe.u32 %0, %1, %2, %3;" : "=r"(result) : "r"(val), "r"(pos), "r"(len));
    return result;
}

__device__ __forceinline__ unsigned int device_bfi(unsigned int base, unsigned int insert, unsigned int pos, unsigned int len) {
    unsigned int result;
    asm("bfi.b32 %0, %1, %2, %3, %4;" : "=r"(result) : "r"(insert), "r"(base), "r"(pos), "r"(len));
    return result;
}

extern "C" __global__
void ISA5b_bfe_bfi(unsigned int* out, const unsigned int* in, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        unsigned int val = in[tid];
        // Extract bits [15:8] (byte 1)
        unsigned int extracted = device_bfe(val, 8, 8);
        // Insert that byte into position [23:16] of 0
        unsigned int inserted = device_bfi(0u, extracted, 16, 8);
        out[tid] = inserted;
    }
}
