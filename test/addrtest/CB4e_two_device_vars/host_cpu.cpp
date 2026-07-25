#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4e_two_device_vars();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);
    memset(h_out, 0, N * sizeof(float));

    // __device__ int d_offset = 10;      -> at c[4][0x0]
    // __device__ float d_factor = 3.0f;  -> at c[4][0x8]
    // SASS: LDC.64 R2, c[0x4][RZ]    -> load pointer from c[4][0..7]
    //       LDG.E R2, desc[UR4][R2.64]  -> dereference to get d_offset (int)
    //       LDC.64 R6, c[0x4][0x8]   -> load pointer from c[4][8..15]
    //       LDG.E R6, desc[UR4][R6.64]  -> dereference to get d_factor (float)
    int *d_offset_ptr = new int(10);
    float *d_factor_ptr = new float(3.0f);
    uint64_t addr0 = (uint64_t)(uintptr_t)d_offset_ptr;
    uint64_t addr1 = (uint64_t)(uintptr_t)d_factor_ptr;
    memcpy(&const_mem[4][0], &addr0, sizeof(uint64_t));   // c[4][0x0]
    memcpy(&const_mem[4][8], &addr1, sizeof(uint64_t));   // c[4][0x8]

    launchKernel(CB4e_two_device_vars, (N+255)/256, 256, h_in, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int idx = i + 10;
        float expected = (idx < N) ? (float)(idx + 1) * 3.0f : 0.0f;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_offset_ptr;
    delete d_factor_ptr;
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
