#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4g_device_zeroinit();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    int *h_out = new int[N];
    memset(h_out, 0xFF, N * sizeof(int));

    // __device__ int d_zero;  (not initialized — CUDA zero-initializes)
    // SASS: LDC.64 R2, c[0x4][RZ]  -> load pointer from c[4][0..7]
    //       LDG.E R4, [R2]          -> dereference to get d_zero (should be 0)
    // c[4][0..7] = pointer to d_zero
    int *d_zero_ptr = new int(0);  // zero-initialized like CUDA
    uint64_t addr = (uint64_t)(uintptr_t)d_zero_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4g_device_zeroinit, (N+255)/256, 256, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != i) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << i << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_zero_ptr;
    delete[] h_out;
    return ok ? 0 : 1;
}
