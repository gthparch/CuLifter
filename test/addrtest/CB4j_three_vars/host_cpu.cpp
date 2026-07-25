#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void CB4j_three_vars();

// Access const_mem to populate c[4] with __device__ variable pointers
extern "C" alignas(8) thread_local uint8_t const_mem[5][4096];

int main() {
    constexpr int N = 256;
    float *h_in = new float[N];
    float *h_out = new float[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);
    memset(h_out, 0, N * sizeof(float));

    // __device__ int d_count = 100;        -> at c[4][0x0]
    // __device__ float d_scale = 0.5f;     -> at c[4][0x8]
    // __device__ double d_bias = 3.14159;  -> at c[4][0x10]
    // SASS: LDC.64 R2, c[0x4][RZ]         -> load pointer from c[4][0..7]  (d_count)
    //       LDG.E R2, desc[UR4][R2.64]     -> dereference to get d_count (int)
    //       LDC.64 R4, c[0x4][0x8]         -> load pointer from c[4][8..15] (d_scale)
    //       LDG.E R4, desc[UR4][R4.64]     -> dereference to get d_scale (float)
    //       LDC.64 R6, c[0x4][0x10]        -> load pointer from c[4][16..23] (d_bias)
    //       LDG.E.64 R6, desc[UR4][R6.64]  -> dereference to get d_bias (double)
    int *d_count_ptr = new int(100);
    float *d_scale_ptr = new float(0.5f);
    double *d_bias_ptr = new double(3.14159);
    uint64_t addr0 = (uint64_t)(uintptr_t)d_count_ptr;
    uint64_t addr1 = (uint64_t)(uintptr_t)d_scale_ptr;
    uint64_t addr2 = (uint64_t)(uintptr_t)d_bias_ptr;
    memcpy(&const_mem[4][0],    &addr0, sizeof(uint64_t));  // c[4][0x0]
    memcpy(&const_mem[4][8],    &addr1, sizeof(uint64_t));  // c[4][0x8]
    memcpy(&const_mem[4][0x10], &addr2, sizeof(uint64_t));  // c[4][0x10]

    launchKernel(CB4j_three_vars, (N+255)/256, 256, h_in, h_out, N);

    bool ok = true;
    for (int i = 0; i < 100; i++) {
        float expected = (float)(i + 1) * 0.5f + (float)3.14159;
        if (fabsf(h_out[i] - expected) > 1e-3f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " expected " << expected << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete d_count_ptr;
    delete d_scale_ptr;
    delete d_bias_ptr;
    delete[] h_in;
    delete[] h_out;
    return ok ? 0 : 1;
}
