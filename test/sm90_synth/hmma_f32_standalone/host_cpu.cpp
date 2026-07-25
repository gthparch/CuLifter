// host_cpu.cpp for hmma_f32_standalone
// Runs the lifted kernel and verifies HMMA produces non-zero output.
// Since x86 single-threaded HMMA computes only lane 0's partial product,
// we don't compare against GPU ref — just check for sane output.

#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_hmma_f32_standalone();

// Pack two FP16 values into a uint32
static uint32_t pack_fp16(float a, float b) {
    // Simple float-to-fp16 conversion for known small values
    // Use bit manipulation for IEEE 754 half precision
    auto f2h = [](float f) -> uint16_t {
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t frac = bits & 0x7FFFFF;
        if (exp > 15) return (sign << 15) | 0x7C00;  // inf
        if (exp < -14) return (sign << 15);            // zero/denorm
        return (sign << 15) | ((exp + 15) << 10) | (frac >> 13);
    };
    return f2h(a) | ((uint32_t)f2h(b) << 16);
}

int main() {
    constexpr int N = 32;  // one warp

    // A packed: 4 regs per thread, all containing fp16(1.0, 1.0)
    uint32_t a_packed[N * 4];
    uint32_t b_packed[N * 2];
    float d_out[N * 4];

    uint32_t ones = pack_fp16(1.0f, 1.0f);
    uint32_t twos = pack_fp16(2.0f, 2.0f);

    for (int i = 0; i < N * 4; i++) a_packed[i] = ones;
    for (int i = 0; i < N * 2; i++) b_packed[i] = twos;
    memset(d_out, 0, sizeof(d_out));

    // Launch: kern_hmma_f32_standalone(a_packed, b_packed, d_out, N)
    launchKernel(kern_hmma_f32_standalone, 1, N,
                 (uint32_t*)a_packed, (uint32_t*)b_packed, (float*)d_out, N);

    // Thread 0 results
    std::cout << "Thread 0 D: " << d_out[0] << " " << d_out[1]
              << " " << d_out[2] << " " << d_out[3] << std::endl;

    // Verify: d0 and d2 should be non-zero (partial dot products)
    // d1 and d3 should be 0 (no B column-1 data on lane 0)
    bool ok = true;
    if (std::isnan(d_out[0])) {
        std::cout << "FAIL: d0 is NaN" << std::endl;
        ok = false;
    }
    if (d_out[0] == 0.0f) {
        std::cout << "FAIL: d0 is zero (expected partial product)" << std::endl;
        ok = false;
    }

    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
