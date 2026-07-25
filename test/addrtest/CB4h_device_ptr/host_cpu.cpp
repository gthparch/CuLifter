#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4h_device_ptr();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    float *h_out = new float[N];
    memset(h_out, 0, N * sizeof(float));

    // __device__ float* d_ptr;
    // This is double indirection:
    //   c[4][0..7] = pointer to d_ptr (a float**)
    //   d_ptr itself points to a float array
    // SASS: LDC.64 R2, c[0x4][RZ]       -> load ptr-to-d_ptr from c[4][0..7]
    //       LDG.E.64 R2, desc[UR4][R2.64] -> dereference to get d_ptr (float*)
    //       IMAD.WIDE R4, R9, 0x4, R2    -> &d_ptr[i]
    //       LDG.E R5, desc[UR4][R4.64]   -> load d_ptr[i]

    // First, allocate the source float array
    float *h_src = new float[N];
    for (int i = 0; i < N; i++) h_src[i] = (float)(i * 7 + 3);

    // d_ptr is a __device__ float* — allocate it on the heap
    // It should hold a pointer to h_src
    float **d_ptr_storage = new float*(h_src);

    // c[4][0..7] = address of d_ptr_storage (which contains the float* to h_src)
    uint64_t addr = (uint64_t)(uintptr_t)d_ptr_storage;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4h_device_ptr, (N+255)/256, 256, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i * 7 + 3);
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_ptr_storage;
    delete[] h_src;
    delete[] h_out;
    return ok ? 0 : 1;
}
