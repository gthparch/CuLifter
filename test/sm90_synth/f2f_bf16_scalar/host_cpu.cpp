#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void kern_f2f_bf16_scalar();

// Exact host conversions (IEEE RN-even / RZ; bf16 = f32's top 16 bits).
// Native-H100 validation pinned cvt.rn.bf16.f32(0x7FC00000) == 0x7FFF:
// sign and payload are dropped.
static uint16_t f32_to_bf16_rn(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if (std::isnan(f)) return (uint16_t)0x7FFF;
    uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7FFF + lsb) >> 16);
}
static uint16_t f32_to_bf16_rz(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if (std::isnan(f)) return (uint16_t)0x7FFF;
    return (uint16_t)(u >> 16);
}

static void fill_inputs(float* h_in, int n) {
    uint32_t pats[] = {
        0x3F800000u, 0xBF800000u,
        0x3F808000u, 0x3F818000u,
        0x3F807FFFu, 0x3F818001u,
        0x7F7F8000u, 0x7F7FFFFFu,
        0x00008000u, 0x00018000u, 0x807FFFFFu,
        0x7F800000u, 0xFF800000u,
        0x7FC00000u, 0xFFC00001u,
        0x7F800001u, 0xFF80FFFFu,
        0x00000000u, 0x80000000u,
    };
    int npat = sizeof(pats) / sizeof(pats[0]);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        if (i < npat) u = pats[i];
        else u = (0x30000000u + 0x01010101u * (uint32_t)i) ^ ((uint32_t)i << 13);
        std::memcpy(&h_in[i], &u, 4);
    }
}

int main() {
    constexpr int N = 128;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    float *in_arr = new float[N];
    uint16_t *rn_arr = new uint16_t[N];
    uint16_t *rz_arr = new uint16_t[N];
    float *widen_arr = new float[N];
    fill_inputs(in_arr, N);
    std::memset(rn_arr, 0, N * sizeof(uint16_t));
    std::memset(rz_arr, 0, N * sizeof(uint16_t));
    std::memset(widen_arr, 0, N * sizeof(float));

    launchKernel(kern_f2f_bf16_scalar, gridDim, blockDim,
                 (const float*)in_arr, rn_arr, rz_arr, widen_arr, N);

    // The scalar F2F.BF16.F32 (RN) / F2FP...PACK_AB.RZ results and the
    // SHF.L 0x10 widening must match the exact host conversions. The
    // kernel's GPU main() asserts the same expectations on hardware.
    int n_fail = 0, first = -1;
    for (int i = 0; i < N; i++) {
        uint16_t ern = f32_to_bf16_rn(in_arr[i]);
        uint16_t erz = f32_to_bf16_rz(in_arr[i]);
        uint32_t ew32 = (uint32_t)ern << 16;
        uint32_t gw32; std::memcpy(&gw32, &widen_arr[i], 4);
        if (rn_arr[i] != ern || rz_arr[i] != erz || gw32 != ew32) {
            if (first < 0) first = i;
            n_fail++;
        }
    }

    bool ok = (n_fail == 0);
    if (!ok) {
        uint32_t u; std::memcpy(&u, &in_arr[first], 4);
        std::cout << "FAIL: " << n_fail << "/" << N
                  << " lanes differ; first at " << first << std::hex
                  << " in=0x" << u
                  << " rn got=0x" << rn_arr[first]
                  << " want=0x" << f32_to_bf16_rn(in_arr[first])
                  << " rz got=0x" << rz_arr[first]
                  << " want=0x" << f32_to_bf16_rz(in_arr[first])
                  << std::dec << std::endl;
    }
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] rn_arr; delete[] rz_arr; delete[] widen_arr;
    return ok ? 0 : 1;
}
