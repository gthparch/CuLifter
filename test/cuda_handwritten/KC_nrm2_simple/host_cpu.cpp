// KC_nrm2_simple host driver (sm80 native, ARG_BASE=0x160).
//
// Runs both kernels in sequence:
//   L0: kc_nrm2_partial → workspace[blockIdx] = sum of squares per CTA
//   L1: kc_nrm2_final   → result = sqrt(sum of partials)
//
// Shadows CB_snrm2's two-launch shape but with kernels we control.
// PASS here means the lifter handles SMEM-reduce + cross-launch workspace.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kc_nrm2_partial();
extern "C" void kc_nrm2_final();

int main() {
    constexpr int N = 1024;          // matches CB_snrm2's N
    constexpr int BLOCK = 128;       // matches CB_snrm2's blockDim
    constexpr int GRID = (N + BLOCK - 1) / BLOCK;

    // ---- Buffers ----
    float* hx       = (float*)std::aligned_alloc(64, N * sizeof(float));
    float* hpartial = (float*)std::aligned_alloc(64, GRID * sizeof(float));
    float* hresult  = (float*)std::aligned_alloc(64, sizeof(float));

    // Match CB_snrm2 init formula
    for (int i = 0; i < N; ++i) hx[i] = (float)((i % 17 + 1) * 0.1f);

    // ---- CPU reference ----
    double sum_sq = 0.0;
    for (int i = 0; i < N; ++i) sum_sq += (double)hx[i] * (double)hx[i];
    float ref = (float)std::sqrt(sum_sq);

    constexpr int BASE = SLIFTER_ARG_BASE;

    // ===== Launch 0: partial reduction =====
    // KCNrm2PartialParams { const float* x, float* partial, int n; }
    //   +0x00 x   (8B)
    //   +0x08 partial (8B)
    //   +0x10 n   (4B)
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hx);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hpartial);
    writeStructField<int32_t> (BASE + 0x10, N);

    fprintf(stderr, "[L0] grid=%d block=%d N=%d\n", GRID, BLOCK, N);
    launchKernel(kc_nrm2_partial, GRID, BLOCK);

    // ---- Mid-launch sanity: did L0 actually write the workspace? ----
    int n_nonzero_partials = 0;
    float partial_sum = 0.0f;
    for (int i = 0; i < GRID; ++i) {
        if (hpartial[i] != 0.0f) ++n_nonzero_partials;
        partial_sum += hpartial[i];
    }
    fprintf(stderr, "[L0 dbg] %d/%d partials non-zero, sum=%.6f (ref=%.6f)\n",
            n_nonzero_partials, GRID, partial_sum, (float)sum_sq);

    // ===== Launch 1: final reduction =====
    // KCNrm2FinalParams { const float* partial, float* result, int n_partials; }
    writeStructField<uint64_t>(BASE + 0x00, (uint64_t)(uintptr_t)hpartial);
    writeStructField<uint64_t>(BASE + 0x08, (uint64_t)(uintptr_t)hresult);
    writeStructField<int32_t> (BASE + 0x10, GRID);

    fprintf(stderr, "[L1] grid=1 block=%d n_partials=%d\n", BLOCK, GRID);
    launchKernel(kc_nrm2_final, 1, BLOCK);

    // ---- Diff ----
    float got = *hresult;
    float diff = std::fabs(got - ref);
    bool pass = (diff / std::max(std::fabs(ref), 1.0f)) < 1e-3f;
    if (pass) {
        std::cout << "TEST PASSED  got=" << got << " ref=" << ref
                  << " diff=" << diff << std::endl;
    } else {
        std::cout << "FAIL: got=" << got << " ref=" << ref
                  << " diff=" << diff << std::endl;
    }

    std::free(hx); std::free(hpartial); std::free(hresult);
    return pass ? 0 : 1;
}
