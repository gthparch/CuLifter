// host_cpu.cpp — driver for KC_gemv_n_simple (sm80 native ABI, ARG_BASE=0x160)
//
// y(M) = alpha * A(MxN) * x(N) + beta * y(M).  A col-major.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_gemv_n_simple();

int main() {
    constexpr int M = 64, N = 32;
    constexpr float alpha = 1.5f, beta = 0.5f;

    float* hA = (float*)std::aligned_alloc(64, M * N * sizeof(float));
    float* hx = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hy = (float*)std::aligned_alloc(64, M * sizeof(float));
    float* hy_ref = (float*)std::aligned_alloc(64, M * sizeof(float));

    for (int k = 0; k < M * N; ++k) hA[k] = (float)((k % 17 + 1) * 0.01f);
    for (int k = 0; k < N; ++k)     hx[k] = (float)((k % 13 + 1) * 0.01f);
    for (int k = 0; k < M; ++k)     hy[k] = (float)((k %  7 + 1) * 0.10f);

    for (int i = 0; i < M; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += hA[i + j * M] * hx[j];
        hy_ref[i] = alpha * acc + beta * hy[i];
    }

    // Struct layout: A(8) x(8) y(8) alpha(4) beta(4) M(4) N(4)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hA);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hx);
    writeStructField<uint64_t>(BASE + 0x10, (uint64_t)(uintptr_t)hy);
    writeStructField<float>   (BASE + 0x18, alpha);
    writeStructField<float>   (BASE + 0x1c, beta);
    writeStructField<int32_t> (BASE + 0x20, M);
    writeStructField<int32_t> (BASE + 0x24, N);

    constexpr int BLOCK = 64;
    int grid = (M + BLOCK - 1) / BLOCK;
    fprintf(stderr, "[KC_gemv_n_simple] M=%d N=%d alpha=%.2f beta=%.2f grid=%d block=%d\n",
            M, N, alpha, beta, grid, BLOCK);
    launchKernel(kc_gemv_n_simple, grid, BLOCK);

    int n_fail = 0, first_fail = -1;
    float max_diff = 0.0f;
    for (int i = 0; i < M; ++i) {
        float d = std::fabs(hy[i] - hy_ref[i]);
        float scale = std::max(std::fabs(hy_ref[i]), 1.0f);
        if (d / scale > 1e-3f) {
            if (first_fail < 0) first_fail = i;
            n_fail++;
        }
        if (d > max_diff) max_diff = d;
    }
    int rc = 1;
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (max_diff=" << max_diff << ", n=" << M << ")"
                  << std::endl;
        rc = 0;
    } else {
        std::cout << "FAIL: " << n_fail << "/" << M << " differ; first at "
                  << first_fail << " got=" << hy[first_fail] << " ref="
                  << hy_ref[first_fail] << " max_diff=" << max_diff << std::endl;
    }
    std::free(hA); std::free(hx); std::free(hy); std::free(hy_ref);
    return rc;
}
