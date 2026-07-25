// KC_axpy host driver — sanity test for the shadow-kernel pipeline.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_axpy();

int main() {
    constexpr int N = 1024;
    constexpr float alpha = 2.5f;

    float* hx = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hy = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hy_ref = (float*)std::aligned_alloc(64, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        hx[i] = (float)((i % 17 + 1) * 0.1f);
        hy[i] = (float)((i % 13 + 1) * 0.2f);
        hy_ref[i] = alpha * hx[i] + hy[i];
    }

    // Struct layout (natural alignment): y(8) x(8) alpha(4) n(4)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hy);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hx);
    writeStructField<float>   (BASE + 0x10, alpha);
    writeStructField<int32_t> (BASE + 0x14, N);

    constexpr int BLOCK = 256;
    int grid = (N + BLOCK - 1) / BLOCK;
    fprintf(stderr, "[KC_axpy] N=%d alpha=%.2f grid=%d block=%d\n",
            N, alpha, grid, BLOCK);

    // Debug: dump bytes at 0x160-0x180
    fprintf(stderr, "[dbg] const_mem[0][0x160..0x180]:\n");
    for (int o = SLIFTER_ARG_BASE; o < SLIFTER_ARG_BASE + 0x20; o += 8) {
        uint64_t v;
        std::memcpy(&v, &const_mem[0][o], 8);
        fprintf(stderr, "  +0x%03x = 0x%016lx\n", o, v);
    }
    float a_check; std::memcpy(&a_check, &const_mem[0][0x170], 4);
    int32_t n_check; std::memcpy(&n_check, &const_mem[0][0x174], 4);
    fprintf(stderr, "[dbg]  as alpha(float)=%.4f  n(int)=%d\n", a_check, n_check);

    launchKernel(kc_axpy, grid, BLOCK);
    fprintf(stderr, "[dbg] after launch: hy[0]=%.4f hy[1]=%.4f\n", hy[0], hy[1]);

    int n_fail = 0, first_fail = -1;
    float max_diff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float d = std::fabs(hy[i] - hy_ref[i]);
        if (d > 1e-4f) {
            if (first_fail < 0) first_fail = i;
            n_fail++;
        }
        if (d > max_diff) max_diff = d;
    }
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (max_diff=" << max_diff << ", n=" << N << ")"
                  << std::endl;
    } else {
        std::cout << "FAIL: " << n_fail << "/" << N
                  << " differ; first at " << first_fail
                  << " got=" << hy[first_fail]
                  << " ref=" << hy_ref[first_fail] << std::endl;
    }
    std::free(hx); std::free(hy); std::free(hy_ref);
    return n_fail == 0 ? 0 : 1;
}
