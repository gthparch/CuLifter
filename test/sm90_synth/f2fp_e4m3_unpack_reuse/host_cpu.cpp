#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_f2fp_e4m3_unpack();

// Exact e4m3 -> f16 bit conversion (OCP FP8 E4M3: bias 7, no Inf,
// S.1111.111 = NaN; every e4m3 value is exactly representable in f16).
static uint16_t e4m3_to_f16(uint8_t b) {
    uint16_t s = (uint16_t)(b & 0x80) << 8;
    uint32_t e = (b >> 3) & 0xF;
    uint32_t m = b & 0x7;
    if (e == 0) {
        if (m == 0) return s;
        // subnormal: m * 2^-9 -> f16 normal
        int p = (m >= 4) ? 2 : (m >= 2 ? 1 : 0);
        uint16_t exp_f = (uint16_t)(p + 6) << 10;
        uint16_t man_f = (uint16_t)((m ^ (1u << p)) << (10 - p));
        return s | exp_f | man_f;
    }
    if (e == 0xF && m == 0x7) return s | 0x7E00;  // NaN
    return s | (uint16_t)((e + 8) << 10) | (uint16_t)(m << 7);
}

int main() {
    constexpr int N = 64;
    uint32_t in_arr[N], out_arr[N * 2];
    for (int i = 0; i < N; i++) {
        uint32_t b0 = 4*i, b1 = 4*i+1, b2 = 4*i+2, b3 = 4*i+3;
        in_arr[i] = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    memset(out_arr, 0, sizeof(out_arr));

    launchKernel(kern_f2fp_e4m3_unpack, 1, 64,
                 (const uint32_t*)in_arr, out_arr, N);

    int n_fail = 0, first = -1;
    for (int i = 0; i < N; i++) {
        for (int w = 0; w < 2; w++) {
            uint8_t blo = (uint8_t)(in_arr[i] >> (16 * w));
            uint8_t bhi = (uint8_t)(in_arr[i] >> (16 * w + 8));
            uint32_t exp = (uint32_t)e4m3_to_f16(blo)
                         | ((uint32_t)e4m3_to_f16(bhi) << 16);
            uint32_t got = out_arr[i * 2 + w];
            if (got != exp) {
                if (first < 0) first = i * 2 + w;
                if (n_fail < 4)
                    printf("MISMATCH word %d (bytes %02x %02x): got=0x%08x exp=0x%08x\n",
                           i * 2 + w, blo, bhi, got, exp);
                n_fail++;
            }
        }
    }
    if (n_fail) {
        std::cout << "TEST FAILED (" << n_fail << "/128 words)" << std::endl;
        return 1;
    }
    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
