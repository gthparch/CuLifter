#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4i_large_array();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 1024;
    float *h_out = new float[N];
    memset(h_out, 0, N * sizeof(float));

    // __device__ float d_lut[1024];
    // SASS: LDC.64 R2, c[0x4][RZ]       -> load pointer from c[4][0..7]
    //       IMAD.WIDE R2, R7, 0x4, R2    -> &d_lut[i]
    //       LDG.E R2, desc[UR4][R2.64]   -> load d_lut[i]
    // Initialize d_lut with pattern: d_lut[i] = i*3+7 (matches ref_host.cu)
    float *d_lut_ptr = new float[N];
    for (int i = 0; i < N; i++) d_lut_ptr[i] = (float)(i * 3 + 7);
    uint64_t addr = (uint64_t)(uintptr_t)d_lut_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4i_large_array, (N+255)/256, 256, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i * 3 + 7) + 1.0f;
        if (fabsf(h_out[i] - expected) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] d_lut_ptr;
    delete[] h_out;
    return ok ? 0 : 1;
}
