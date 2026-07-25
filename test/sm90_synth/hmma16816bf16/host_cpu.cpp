#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the HMMA.16816.F32.BF16 fixture: one warp computes
// D(f32) = A(16x16 bf16) * B(16x8 bf16) + C(f32); host replays in f32
// (bf16 products are exact in f32) and compares within f32 accumulation
// tolerance.
// Param offsets (sm90 base 0x210): A 0x210, B 0x218, C 0x220, D 0x228.

extern "C" void kern_hmma16816bf16();

// f32 -> bf16 bits, round-to-nearest-even (matches cvt.rn.bf16.f32).
static uint16_t f2bf(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    if ((x & 0x7fffffff) > 0x7f800000) return (uint16_t)((x >> 16) | 0x40);  // NaN
    uint32_t rounding = 0x7fff + ((x >> 16) & 1);
    return (uint16_t)((x + rounding) >> 16);
}
// bf16 bits -> f32 (exact: bf16 is the top 16 bits of an f32).
static float bf2f(uint16_t h) {
    uint32_t out = (uint32_t)h << 16;
    float f; std::memcpy(&f, &out, 4); return f;
}

int main() {
    constexpr int M = 16, N = 8, K = 16;
    std::vector<uint16_t> A(M * K), B(K * N);
    std::vector<float> C(M * N), D(M * N, -7.0f);
    for (int i = 0; i < M * K; ++i) A[i] = f2bf(0.125f * ((i % 11) + 1) - 0.75f);
    for (int i = 0; i < K * N; ++i) B[i] = f2bf(0.0625f * ((i % 13) + 1) - 0.4375f);
    for (int i = 0; i < M * N; ++i) C[i] = 0.25f * ((i % 7) + 1);

    launchKernel(kern_hmma16816bf16, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data());

    int nbad = 0; float maxd = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = C[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += bf2f(A[m * K + k]) * bf2f(B[k * N + n]);
            float exp = acc;
            float got = D[m * N + n];
            float diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            // bf16 products are exact in f32; only the 16-term accumulation
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
