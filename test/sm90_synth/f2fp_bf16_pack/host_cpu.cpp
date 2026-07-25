#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_f2fp_bf16_pack();

// f32 -> bf16 round-to-nearest-even (bf16 = f32's top 16 bits)
static uint16_t f32_to_bf16_rn(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if (std::isnan(f)) return (uint16_t)((u >> 16) | 0x0040);
    uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7FFF + lsb) >> 16);
}

int main() {
    constexpr int N = 64;
    float in_arr[N * 2];
    uint32_t out_arr[N];
    for (int i = 0; i < N * 2; i++) {
        float v = (float)((i % 23) + 1) * 0.731f;
        int e = (i % 17) - 8;
        in_arr[i] = ((i & 1) ? -1.0f : 1.0f) * std::ldexp(v, e * 3);
        if (i == 5) in_arr[i] = 14.75f;
        if (i == 9) in_arr[i] = 1.00390625f;
    }
    memset(out_arr, 0, sizeof(out_arr));

    launchKernel(kern_f2fp_bf16_pack, 1, 64,
                 (const float*)in_arr, out_arr, N);

    int n_fail = 0;
    for (int i = 0; i < N; i++) {
        uint32_t exp = ((uint32_t)f32_to_bf16_rn(in_arr[i * 2]) << 16)
                     | f32_to_bf16_rn(in_arr[i * 2 + 1]);
        if (out_arr[i] != exp) {
            if (n_fail < 4)
                printf("MISMATCH %d (a=%g b=%g): got=0x%08x exp=0x%08x\n",
                       i, in_arr[i*2], in_arr[i*2+1], out_arr[i], exp);
            n_fail++;
        }
    }
    if (n_fail) {
        std::cout << "TEST FAILED (" << n_fail << "/64)" << std::endl;
        return 1;
    }
    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
