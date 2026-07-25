#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the HMMA.1688.F32.TF32 fixture: one warp computes
// D(f32) = A(16x8 tf32) * B(8x8 tf32) + C(f32). Inputs are RANDOM signed
// f32 values pre-snapped to the tf32 grid (low 13 mantissa bits zeroed),
// matching the cvt-converted operand domain of real CUTLASS/cuBLAS TF32
// kernels; tf32 products are then exact in f32, so only the 8-term
// accumulation order can differ.
// Param offsets (sm90 base 0x210): A 0x210, B 0x218, C 0x220, D 0x228.

extern "C" void kern_hmma1688tf32();

// f32 -> nearest tf32-grid value (truncate low 13 mantissa bits).
static float snap_tf32(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    x &= 0xFFFFE000u;
    float r; std::memcpy(&r, &x, 4); return r;
}

int main() {
    constexpr int M = 16, N = 8, K = 8;
    std::vector<float> A(M * K), B(K * N), C(M * N), D(M * N, -7.0f);
    // Deterministic LCG; random SIGNED values (constant matrices are how
    // layout bugs hide).
    uint32_t s = 0x1688F32u;
    auto nxt = [&]() { s = s * 1664525u + 1013904223u; return s; };
    auto rnd = [&](float scale) {
        return scale * (((int)(nxt() >> 16 & 0x3FF) - 512) / 256.0f);
    };
    for (int i = 0; i < M * K; ++i) A[i] = snap_tf32(rnd(1.0f));
    for (int i = 0; i < K * N; ++i) B[i] = snap_tf32(rnd(1.0f));
    for (int i = 0; i < M * N; ++i) C[i] = rnd(2.0f);

    launchKernel(kern_hmma1688tf32, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data());

    int nbad = 0; float maxd = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = C[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += A[m * K + k] * B[k * N + n];
            float exp = acc;
            float got = D[m * N + n];
            float diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            // tf32 products are exact in f32; only the 8-term accumulation
            // order can differ (HW adds in pairs) — ulp-scale slack.
            if (diff > 1e-3f * std::max(std::fabs(exp), 1.0f)) {
                if (nbad < 4) printf("bad (%d,%d) got=%g exp=%g\n", m, n, got, exp);
                nbad++;
            }
        }
    if (!nbad) { printf("REF VERIFIED (max_diff=%g)\nTEST PASSED\n", maxd); return 0; }
    printf("FAILED: %d/%d mismatch max_diff=%g\n", nbad, M * N, maxd);
    return 1;
}
