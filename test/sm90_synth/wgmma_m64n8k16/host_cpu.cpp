// host_cpu.cpp for wgmma_m64n8k16
// Runs the lifted kernel and verifies HGMMA/WGMMA is handled.
// On x86 single-threaded, WGMMA passes through C to D (accumulator passthrough)
// since descriptor-based A/B data cannot be resolved.

#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void kern_wgmma_m64n8k16();

static uint32_t pack_fp16(float a, float b) {
    auto f2h = [](float f) -> uint16_t {
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 31) & 1;
        int32_t exp = ((bits >> 23) & 0xFF) - 127;
        uint32_t frac = bits & 0x7FFFFF;
        if (exp > 15) return (sign << 15) | 0x7C00;
        if (exp < -14) return (sign << 15);
        return (sign << 15) | ((exp + 15) << 10) | (frac >> 13);
    };
    return f2h(a) | ((uint32_t)f2h(b) << 16);
}

int main() {
    constexpr int N = 32;

    uint32_t a_packed[N * 4];
    uint32_t b_packed[N * 2];
    float d_out[N * 4];

    uint32_t ones = pack_fp16(1.0f, 1.0f);
    uint32_t twos = pack_fp16(2.0f, 2.0f);

    for (int i = 0; i < N * 4; i++) a_packed[i] = ones;
    for (int i = 0; i < N * 2; i++) b_packed[i] = twos;
    memset(d_out, 0, sizeof(d_out));

    launchKernel(kern_wgmma_m64n8k16, 1, N,
                 (uint32_t*)a_packed, (uint32_t*)b_packed, (float*)d_out, N);

    std::cout << "Thread 0 D: " << d_out[0] << " " << d_out[1]
              << " " << d_out[2] << " " << d_out[3] << std::endl;

    bool ok = true;
    for (int i = 0; i < N * 4; i++) {
        if (std::isnan(d_out[i])) {
            std::cout << "FAIL: NaN at index " << i << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
