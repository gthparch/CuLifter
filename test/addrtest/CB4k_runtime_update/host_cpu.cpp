#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4k_runtime_update();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);
    memset(h_out, 0, N * sizeof(float));

    // __device__ float d_multiplier;  -> at c[4][0x0]
    // SASS: LDC.64 R4, c[0x4][RZ]        -> load pointer from c[4][0..7]
    //       LDG.E R4, desc[UR4][R4.64]    -> dereference to get d_multiplier (float)
    // Runtime value: 7.5f (matches ref_host.cu)
    float *d_mult_ptr = new float(7.5f);
    uint64_t addr = (uint64_t)(uintptr_t)d_mult_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4k_runtime_update, (N+255)/256, 256, h_in, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = (float)(i + 1) * 7.5f;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_mult_ptr;
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
