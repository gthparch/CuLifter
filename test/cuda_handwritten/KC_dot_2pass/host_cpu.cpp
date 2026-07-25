// host_cpu.cpp — driver for KC_dot_2pass (sm80 native ABI, ARG_BASE=0x160)
//
// Two-launch dot product. Per-CTA partials reduced to scalar by 2nd kernel.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_dot_partial();
extern "C" void kc_dot_reduce();

int main() {
    constexpr int N = 1024;
    constexpr int BLOCK = 32;     // single warp per block
    constexpr int GRID  = 8;      // 8 partials

    float* hx = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hy = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hpart = (float*)std::aligned_alloc(64, GRID * sizeof(float));
    float h_result = 0.0f;
    float ref = 0.0f;
    for (int i = 0; i < N; ++i) {
        hx[i] = (float)((i % 11) - 5) * 0.1f;
        hy[i] = (float)((i % 7) - 3) * 0.2f;
        ref += hx[i] * hy[i];
    }
    for (int i = 0; i < GRID; ++i) hpart[i] = 0.0f;

    // Launch 0: KCDotPartialParams { x(8), y(8), partial(8), n(4) }
    {
        constexpr int BASE = SLIFTER_ARG_BASE;
        writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hx);
        writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hy);
        writeStructField<uint64_t>(BASE + 0x10, (uint64_t)(uintptr_t)hpart);
        writeStructField<int32_t> (BASE + 0x18, N);
        fprintf(stderr, "[KC_dot_2pass.partial] N=%d grid=%d block=%d\n",
                N, GRID, BLOCK);
        launchKernel(kc_dot_partial, GRID, BLOCK);
    }

    // Launch 1: KCDotReduceParams { partial(8), result(8), n_partials(4) }
    {
        constexpr int BASE = SLIFTER_ARG_BASE;
        writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hpart);
        writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)&h_result);
        writeStructField<int32_t> (BASE + 0x10, GRID);
        fprintf(stderr, "[KC_dot_2pass.reduce] n_partials=%d block=%d\n",
                GRID, BLOCK);
        launchKernel(kc_dot_reduce, /*grid=*/1, BLOCK);
    }

    float diff = std::fabs(h_result - ref);
    float scale = std::max(std::fabs(ref), 1.0f);
    int rc = 1;
    if (diff / scale < 1e-3f) {
        std::cout << "TEST PASSED  (got=" << h_result << " ref=" << ref
                  << " diff=" << diff << ")" << std::endl;
        rc = 0;
    } else {
        std::cout << "FAIL  got=" << h_result << " ref=" << ref
                  << " diff=" << diff << std::endl;
    }
    std::free(hx); std::free(hy); std::free(hpart);
    return rc;
}
