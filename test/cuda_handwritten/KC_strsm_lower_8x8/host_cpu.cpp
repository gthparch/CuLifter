// host_cpu.cpp — driver for KC_strsm_lower_8x8 (sm80 native ABI, ARG_BASE=0x160)
//
// Solves L * X = alpha * B (X overwrites B). 8x8 col-major lower-triangular L.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_strsm_lower_8x8();

int main() {
    constexpr int N = 8;
    constexpr float alpha = 2.0f;

    float* hL = (float*)std::aligned_alloc(64, N * N * sizeof(float));
    float* hX = (float*)std::aligned_alloc(64, N * N * sizeof(float));
    float* hX_init = (float*)std::aligned_alloc(64, N * N * sizeof(float));
    float* hX_ref = (float*)std::aligned_alloc(64, N * N * sizeof(float));

    // Lower triangular L with diag=1.0; subdiag entries small.
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            hL[i + j * N] = (i == j) ? 1.0f : (i > j ? 0.1f * (float)(i - j) : 0.0f);
        }
    }
    for (int k = 0; k < N * N; ++k) {
        hX_init[k] = (float)((k % 11 + 1) * 0.05f);
        hX[k] = hX_init[k];
    }

    // CPU reference (forward substitution).
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float sum = alpha * hX_init[i + j * N];
            for (int k = 0; k < i; ++k) sum -= hL[i + k * N] * hX_ref[k + j * N];
            hX_ref[i + j * N] = sum / hL[i + i * N];
        }
    }

    // Struct layout: L(8) X(8) alpha(4)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hL);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hX);
    writeStructField<float>   (BASE + 0x10, alpha);

    fprintf(stderr, "[KC_strsm_lower_8x8] N=%d alpha=%.2f block=8\n", N, alpha);
    launchKernel(kc_strsm_lower_8x8, /*grid=*/1, /*block=*/8);

    int n_fail = 0, first_fail = -1;
    float max_diff = 0.0f;
    for (int k = 0; k < N * N; ++k) {
        float d = std::fabs(hX[k] - hX_ref[k]);
        float scale = std::max(std::fabs(hX_ref[k]), 1.0f);
        if (d / scale > 1e-3f) {
            if (first_fail < 0) first_fail = k;
            n_fail++;
        }
        if (d > max_diff) max_diff = d;
    }
    int rc = 1;
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (max_diff=" << max_diff << ", n=" << N*N << ")"
                  << std::endl;
        rc = 0;
    } else {
        std::cout << "FAIL: " << n_fail << "/" << N*N << " differ; first at "
                  << first_fail << " got=" << hX[first_fail] << " ref="
                  << hX_ref[first_fail] << " max_diff=" << max_diff << std::endl;
    }
    std::free(hL); std::free(hX); std::free(hX_init); std::free(hX_ref);
    return rc;
}
