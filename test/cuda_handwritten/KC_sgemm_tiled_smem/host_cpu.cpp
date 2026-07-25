// host_cpu.cpp — driver for KC_sgemm_tiled_smem (sm80 native ABI, ARG_BASE=0x160)
//
// Tiled SGEMM with shared-memory tile loader.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_sgemm_tiled();

int main() {
    constexpr int M = 32, N = 32, K = 32;
    constexpr float alpha = 1.5f, beta = 0.5f;

    float* hA = (float*)std::aligned_alloc(64, M * K * sizeof(float));
    float* hB = (float*)std::aligned_alloc(64, K * N * sizeof(float));
    float* hC = (float*)std::aligned_alloc(64, M * N * sizeof(float));
    float* hC_ref = (float*)std::aligned_alloc(64, M * N * sizeof(float));

    for (int k = 0; k < M * K; ++k) hA[k] = (float)((k % 17 + 1) * 0.01f);
    for (int k = 0; k < K * N; ++k) hB[k] = (float)((k % 13 + 1) * 0.01f);
    for (int k = 0; k < M * N; ++k) hC[k] = (float)((k %  7 + 1) * 0.10f);

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += hA[i + k * M] * hB[k + j * K];
            hC_ref[i + j * M] = alpha * acc + beta * hC[i + j * M];
        }
    }

    // Struct layout: C(8) A(8) B(8) alpha(4) beta(4) M(4) N(4) K(4)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hC);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hA);
    writeStructField<uint64_t>(BASE + 0x10, (uint64_t)(uintptr_t)hB);
    writeStructField<float>   (BASE + 0x18, alpha);
    writeStructField<float>   (BASE + 0x1c, beta);
    writeStructField<int32_t> (BASE + 0x20, M);
    writeStructField<int32_t> (BASE + 0x24, N);
    writeStructField<int32_t> (BASE + 0x28, K);

    constexpr int TILE = 16;
    Dim3 grid_dim { (M + TILE - 1) / TILE, (N + TILE - 1) / TILE, 1 };
    Dim3 block_dim{ TILE, TILE, 1 };
    fprintf(stderr, "[KC_sgemm_tiled_smem] M=%d N=%d K=%d alpha=%.2f beta=%.2f "
                    "grid=(%d,%d) block=(%d,%d)\n",
            M, N, K, alpha, beta,
            grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    launchKernel2D(kc_sgemm_tiled, grid_dim, block_dim);

    int n_fail = 0, first_fail = -1;
    float max_diff = 0.0f;
    for (int k = 0; k < M * N; ++k) {
        float d = std::fabs(hC[k] - hC_ref[k]);
        float scale = std::max(std::fabs(hC_ref[k]), 1.0f);
        if (d / scale > 1e-3f) {
            if (first_fail < 0) first_fail = k;
            n_fail++;
        }
        if (d > max_diff) max_diff = d;
    }
    int rc = 1;
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (max_diff=" << max_diff << ", n=" << M*N << ")"
                  << std::endl;
        rc = 0;
    } else {
        std::cout << "FAIL: " << n_fail << "/" << M*N << " differ; first at "
                  << first_fail << " got=" << hC[first_fail] << " ref="
                  << hC_ref[first_fail] << " max_diff=" << max_diff << std::endl;
    }
    std::free(hA); std::free(hB); std::free(hC); std::free(hC_ref);
    return rc;
}
