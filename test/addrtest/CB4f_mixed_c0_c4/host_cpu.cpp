#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4f_mixed_c0_c4();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    float scale = 2.0f;
    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.5f;
    memset(h_out, 0, N * sizeof(float));

    // __device__ float d_bias = 100.0f;
    // SASS: LDC.64 R4, c[0x4][RZ]  -> load pointer from c[4][0..7]
    //       LDG.E reads d_bias through that pointer
    // Kernel params (in, out, N, scale) come from c[0] as usual
    // c[4][0..7] = pointer to d_bias
    float *d_bias_ptr = new float(100.0f);
    uint64_t addr = (uint64_t)(uintptr_t)d_bias_ptr;
    memcpy(&const_mem[4][0], &addr, sizeof(uint64_t));

    launchKernel(CB4f_mixed_c0_c4, (N+255)/256, 256, h_in, h_out, N, scale);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = h_in[i] * scale + 100.0f;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_bias_ptr;
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
