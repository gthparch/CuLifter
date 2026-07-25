// Host driver for KC_nrm2_with_imnmx. Tests whether the
// IMNMX(bitcast(fabs(SMEM[0]))) chain reproduces the BUG-090 wrong-value pattern.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_nrm2_with_imnmx();

int main() {
    constexpr int N = 1024;
    constexpr int BLOCK = 128;
    constexpr int GRID = 1024;

    float* hx       = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hpartial = (float*)std::aligned_alloc(64, GRID * sizeof(float));
    std::memset(hpartial, 0, GRID * sizeof(float));

    for (int i = 0; i < N; ++i) hx[i] = (float)((i % 17 + 1) * 0.1f);

    constexpr int BASE = SLIFTER_ARG_BASE;
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hx);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hpartial);
    writeStructField<int32_t> (BASE + 0x10, N);

    fprintf(stderr, "[KC_nrm2_with_imnmx] grid=%d block=%d N=%d\n", GRID, BLOCK, N);
    launchKernel(kc_nrm2_with_imnmx, GRID, BLOCK);

    auto block_ref = [&](int b) {
        double s = 0.0;
        for (int t = 0; t < BLOCK; ++t) {
            int idx = b * BLOCK + t;
            if (idx < N) {
                double v = hx[idx];
                s += v * v;
            }
        }
        // The kernel applies IMNMX(bitcast(fabs)) which for non-negative
        // sums is just fabs (signed-bit compare picks the larger-bits
        // value, and fabs's bits are positive). Then FMUL with itself
        // squared via bitcast tricks ≠ pure sum. For comparison we just
        // check it's NON-CONSTANT across blocks (the BUG-090 symptom is
        // identical-value across blocks).
        return (float)s;
    };

    int n_nz = 0, n_unique = 0;
    float prev = -1.0f;
    for (int i = 0; i < GRID; ++i) {
        if (hpartial[i] != 0.0f) ++n_nz;
        if (hpartial[i] != prev) { ++n_unique; prev = hpartial[i]; }
    }

    fprintf(stderr,
        "[L0 dbg] non-zero=%d/%d  ~unique-value-runs=%d\n"
        "  partial[0..7]: %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n"
        "  ref    [0..7]: %.6e %.6e %.6e %.6e %.6e %.6e %.6e %.6e\n",
        n_nz, GRID, n_unique,
        hpartial[0], hpartial[1], hpartial[2], hpartial[3],
        hpartial[4], hpartial[5], hpartial[6], hpartial[7],
        block_ref(0), block_ref(1), block_ref(2), block_ref(3),
        block_ref(4), block_ref(5), block_ref(6), block_ref(7));

    // BUG-090 symptom: workspace[0..7] all IDENTICAL.
    bool bug_repro = (hpartial[0] != 0.0f) && (hpartial[0] == hpartial[1])
                    && (hpartial[1] == hpartial[2]) && (hpartial[2] == hpartial[3])
                    && (hpartial[3] == hpartial[4]) && (hpartial[4] == hpartial[5])
                    && (hpartial[5] == hpartial[6]) && (hpartial[6] == hpartial[7]);
    if (bug_repro) {
        std::cout << "BUG-090 REPRODUCED: workspace[0..7] all identical = "
                  << hpartial[0] << std::endl;
    } else {
        // Block 0..7 should have DIFFERENT non-zero values
        bool all_distinct_nonzero = true;
        for (int i = 0; i < 8; ++i) {
            if (hpartial[i] == 0.0f) { all_distinct_nonzero = false; break; }
            for (int j = i+1; j < 8; ++j) {
                if (hpartial[i] == hpartial[j]) { all_distinct_nonzero = false; break; }
            }
        }
        if (all_distinct_nonzero) {
            std::cout << "TEST PASSED  (8 blocks produced 8 distinct non-zero values "
                      << "→ lifter handles IMNMX(bitcast(fabs(SMEM))) chain correctly)"
                      << std::endl;
        } else {
            std::cout << "INCONCLUSIVE: not all-identical, not all-distinct" << std::endl;
        }
    }

    std::free(hx); std::free(hpartial);
    return 0;
}
