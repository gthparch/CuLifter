// host_cpu.cpp — driver for KC_sgemm_naive (sm80 native ABI, ARG_BASE=0x160)
//
// Hand-written shadow of CB_sgemm_nn, but with KNOWN STATUS:
//   - Single struct param, natural alignment
//   - CPU reference is the same triple-loop the kernel computes,
//     so PASS == lifter is correct for this shape.
// If this PASSes but CB_sgemm_nn FAILs, the cuBLAS failure is in the
// workspace / negative-offset territory, NOT a lifter bug.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_sgemm_naive();

int main() {
    // Same shape as one of the small CB_sgemm cases for direct comparability.
    constexpr int M = 64, N = 64, K = 64;
    constexpr float alpha = 1.5f, beta = 0.5f;

    // ---- Host buffers ----
    float* hA = (float*)std::aligned_alloc(64, M * K * sizeof(float));
    float* hB = (float*)std::aligned_alloc(64, K * N * sizeof(float));
    float* hC = (float*)std::aligned_alloc(64, M * N * sizeof(float));

    // Deterministic init.
    for (int k = 0; k < M * K; ++k) hA[k] = (float)((k % 17 + 1) * 0.01f);
    for (int k = 0; k < K * N; ++k) hB[k] = (float)((k % 13 + 1) * 0.01f);
    for (int k = 0; k < M * N; ++k) hC[k] = (float)((k %  7 + 1) * 0.10f);

    // ---- CPU reference (computed BEFORE launch — kernel mutates hC) ----
    float* hC_ref = (float*)std::aligned_alloc(64, M * N * sizeof(float));
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < M; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                acc += hA[i + k * M] * hB[k + j * K];
            }
            hC_ref[i + j * M] = alpha * acc + beta * hC[i + j * M];
        }
    }

    // ---- Pack the kernel's struct param into c[0][0x160+] ----
    // Struct layout (natural alignment):
    //   +0x00  C        (ptr,   8B)
    //   +0x08  A        (ptr,   8B)
    //   +0x10  B        (ptr,   8B)
    //   +0x18  alpha    (float, 4B)
    //   +0x1c  beta     (float, 4B)
    //   +0x20  M        (int32, 4B)
    //   +0x24  N        (int32, 4B)
    //   +0x28  K        (int32, 4B)
    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hC);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hA);
    writeStructField<uint64_t>(BASE + 0x10, (uint64_t)(uintptr_t)hB);
    writeStructField<float>   (BASE + 0x18, alpha);
    writeStructField<float>   (BASE + 0x1c, beta);
    writeStructField<int32_t> (BASE + 0x20, M);
    writeStructField<int32_t> (BASE + 0x24, N);
    writeStructField<int32_t> (BASE + 0x28, K);

    // ---- Launch: grid=(M/8, N/8), block=(8, 8) ----
    constexpr int BM = 8, BN = 8;
    Dim3 grid_dim { (M + BM - 1) / BM, (N + BN - 1) / BN, 1 };
    Dim3 block_dim{ BM, BN, 1 };
    fprintf(stderr, "[KC_sgemm_naive] M=%d N=%d K=%d alpha=%.2f beta=%.2f "
                    "grid=(%d,%d) block=(%d,%d)\n",
            M, N, K, alpha, beta,
            grid_dim.x, grid_dim.y, block_dim.x, block_dim.y);
    launchKernel2D(kc_sgemm_naive, grid_dim, block_dim);

    // ---- Diff ----
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
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (max_diff=" << max_diff << ", n=" << M*N << ")"
                  << std::endl;
    } else {
        std::cout << "FAIL: " << n_fail << "/" << M*N
                  << " elements differ; first at " << first_fail
                  << " got=" << hC[first_fail]
                  << " ref=" << hC_ref[first_fail]
                  << " max_diff=" << max_diff << std::endl;
    }

    std::free(hA); std::free(hB); std::free(hC); std::free(hC_ref);
    return n_fail == 0 ? 0 : 1;
}
