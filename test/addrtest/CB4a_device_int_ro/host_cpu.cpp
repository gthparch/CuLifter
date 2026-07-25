#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4a_device_int_ro();

// Access const_mem to populate c[4] with __device__ variable values
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    int *h_out = new int[N];
    memset(h_out, 0, N * sizeof(int));

    // BUG-039 fix: populate c[4] with ADDRESS of __device__ variable
    // The SASS does: LDC.64 R2, c[0x4][RZ]  (load pointer from c[4])
    //                 LDG.E R4, [R2]          (dereference to get value)
    // So c[4][0..7] must contain a valid 64-bit pointer to the data
    int *d_val_ptr = new int(42);  // allocate device variable on heap
    uint64_t addr = (uint64_t)(uintptr_t)d_val_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4a_device_int_ro, (N+255)/256, 256, h_out, N);
    delete d_val_ptr;

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int expected = 42 + i;
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_out;
    return ok ? 0 : 1;
}
