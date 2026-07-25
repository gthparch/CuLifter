// host_cpu.cpp — driver for KC_asum_simple (sm80 native ABI, ARG_BASE=0x160)
//
// Single-block absolute-sum reduction. Verifies BUG-088 fix on a kernel
// that uses warp shuffle + per-warp shared-mem accumulation.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_asum_simple();

int main() {
    constexpr int N = 1024;

    float* hx = (float*)std::aligned_alloc(64, N * sizeof(float));
    float h_result = 0.0f;
    float ref = 0.0f;
    for (int i = 0; i < N; ++i) {
        hx[i] = (float)((i % 17) - 8) * 0.1f;
        ref += std::fabs(hx[i]);
    }

    // Struct layout (natural alignment): x(8) result(8) n(4)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hx);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)&h_result);
    writeStructField<int32_t> (BASE + 0x10, N);

    constexpr int BLOCK = 256;
    fprintf(stderr, "[KC_asum_simple] N=%d block=%d ref=%.4f\n", N, BLOCK, ref);
    launchKernel(kc_asum_simple, /*grid=*/1, BLOCK);

    float diff = std::fabs(h_result - ref);
    float scale = std::max(std::fabs(ref), 1.0f);
    if (diff / scale < 1e-3f) {
        std::cout << "TEST PASSED  (got=" << h_result << " ref=" << ref
                  << " diff=" << diff << ")" << std::endl;
        std::free(hx);
        return 0;
    } else {
        std::cout << "FAIL  got=" << h_result << " ref=" << ref
                  << " diff=" << diff << std::endl;
        std::free(hx);
        return 1;
    }
}
