#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the HMMA.1688.F16 fixture: one warp computes
// D = A(16x8)*B(8x8) + C in f16. Inputs are RANDOM signed values (constant
// matrices are how layout bugs hide); host replays in f32 with a final f16
// round and compares within f16 tolerance.
// Param offsets (sm90 base 0x210): A 0x210, B 0x218, C 0x220, D 0x228.

extern "C" void kern_hmma1688f16();

static uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t  e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (e >= 31) return (uint16_t)(s | 0x7c00);
    if (e <= 0) return (uint16_t)s;
    uint32_t h = s | (uint32_t)(e << 10) | (m >> 13);
    uint32_t rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}
static float h2f(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t out;
    if (e == 0) { if (!m) out = s; else { e = 127 - 15 + 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; out = s | (e << 23) | (m << 13); } }
    else if (e == 31) out = s | 0x7f800000 | (m << 13);
    else out = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

int main() {
    constexpr int M = 16, N = 8, K = 8;
    std::vector<uint16_t> A(M * K), B(K * N), C(M * N), D(M * N, f2h(-7.0f));
    uint32_t s = 0x168816u;
    auto nxt = [&]() { s = s * 1664525u + 1013904223u; return s; };
    auto rnd = [&](float scale) {
        return scale * (((int)(nxt() >> 16 & 0x3FF) - 512) / 256.0f);
    };
    for (int i = 0; i < M * K; ++i) A[i] = f2h(rnd(1.0f));
    for (int i = 0; i < K * N; ++i) B[i] = f2h(rnd(1.0f));
    for (int i = 0; i < M * N; ++i) C[i] = f2h(rnd(2.0f));

    launchKernel(kern_hmma1688f16, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data());

    int nbad = 0; float maxd = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = h2f(C[m * N + n]);
            for (int k = 0; k < K; ++k)
                acc += h2f(A[m * K + k]) * h2f(B[k * N + n]);
            float exp = h2f(f2h(acc));   // final f16 round
            float got = h2f(D[m * N + n]);
            float diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            // f16 ulp-scale tolerance (per-step rounding differences)
            if (diff > 0.03f * std::max(std::fabs(exp), 1.0f)) {
                if (nbad < 4) printf("bad (%d,%d) got=%g exp=%g\n", m, n, got, exp);
                nbad++;
            }
        }
    if (!nbad) { printf("REF VERIFIED (max_diff=%g)\nTEST PASSED\n", maxd); return 0; }
    printf("FAILED: %d/%d mismatch max_diff=%g\n", nbad, M * N, maxd);
    return 1;
}
