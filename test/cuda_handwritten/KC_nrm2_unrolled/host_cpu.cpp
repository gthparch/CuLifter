// KC_nrm2_unrolled host driver (sm80 native, ARG_BASE=0x160).
// Tests the suspect cuBLAS nrm2_kernel pattern in isolation.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_nrm2_unrolled();

int main() {
    constexpr int N = 1024;
    constexpr int BLOCK = 128;
    constexpr int GRID = 1024;     // mimic CB_snrm2's 1024 blocks (only 8 do work)

    float* hx       = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hpartial = (float*)std::aligned_alloc(64, GRID * sizeof(float));
    std::memset(hpartial, 0, GRID * sizeof(float));

    for (int i = 0; i < N; ++i) hx[i] = (float)((i % 17 + 1) * 0.1f);

    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hx);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hpartial);
    writeStructField<int32_t> (BASE + 0x10, N);

    fprintf(stderr, "[KC_nrm2_unrolled] grid=%d block=%d N=%d\n", GRID, BLOCK, N);
    launchKernel(kc_nrm2_unrolled, GRID, BLOCK);

    // Inspect — same diagnostic as CB_snrm2.
    int n_nz = 0;
    for (int i = 0; i < GRID; ++i) if (hpartial[i] != 0.0f) ++n_nz;

    // CPU expected per-block sum-of-squares for blocks 0..7
    auto block_ref = [&](int b) {
        double s = 0.0;
        for (int t = 0; t < BLOCK; ++t) {
            int idx = b * BLOCK + t;
            if (idx < N) {
                double v = hx[idx];
                s += v * v;
            }
        }
        return (float)s;
    };

    fprintf(stderr,
        "[L0 dbg] non-zero=%d/%d\n"
        "  partial[0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n"
        "  ref    [0..7]: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
        n_nz, GRID,
        hpartial[0], hpartial[1], hpartial[2], hpartial[3],
        hpartial[4], hpartial[5], hpartial[6], hpartial[7],
        block_ref(0), block_ref(1), block_ref(2), block_ref(3),
        block_ref(4), block_ref(5), block_ref(6), block_ref(7));

    int n_fail = 0;
    for (int b = 0; b < 8; ++b) {
        float diff = std::fabs(hpartial[b] - block_ref(b));
        if (diff / std::max(std::fabs(block_ref(b)), 1.0f) > 1e-3f) ++n_fail;
    }
    if (n_fail == 0) {
        std::cout << "TEST PASSED  (8/8 blocks match per-block sum-of-squares)"
                  << std::endl;
    } else {
        std::cout << "FAIL: " << n_fail << "/8 blocks mismatch" << std::endl;
    }

    std::free(hx); std::free(hpartial);
    return n_fail == 0 ? 0 : 1;
}
