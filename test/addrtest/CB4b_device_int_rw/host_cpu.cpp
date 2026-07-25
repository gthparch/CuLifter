#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4b_device_int_rw();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    int *h_out = new int[N];
    memset(h_out, 0, N * sizeof(int));

    // __device__ int d_counter;  (uninitialized — default 0)
    // SASS: LDC.64 R2, c[0x4][RZ]  -> load pointer from c[4][0..7]
    //       STG.E desc[UR4][R2.64], R9  -> store i*2 through pointer
    //       STG.E desc[UR4][R4.64], R9  -> store i*2 to out[i]
    // c[4][0..7] = pointer to d_counter
    int *d_counter_ptr = new int(0);
    uint64_t addr = (uint64_t)(uintptr_t)d_counter_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4b_device_int_rw, (N+255)/256, 256, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int expected = i * 2;
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_counter_ptr;
    delete[] h_out;
    return ok ? 0 : 1;
}
