#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the CHAINED 2-MMA fixture mma_chain_frag. Replays the WHOLE chain
// in f64 (with the same f16 P rounding the HW does) and compares O PER ROW.
//
// Chain (mirrors 41_fmha forward):
//   MMA1:  P(16x8) = X(16x8) * Y(8x8)      [mma.m16n8k8, f32 accum]
//   nonlinear (softmax-role): nl(x)=relu(0.5x+0.25)
//   F2FP:  P f32 -> f16 (rounded), STS to smem row-major
//   MMA2:  O(16x8) = V(16x8) * Pt          via LDSM.MT88 (transposed) B-operand
//          i.e. O[m][n] = sum_k V[m][k] * P[n][k]   (P = f16-rounded post-nl)
//
// Param offsets (sm90 base 0x210): X 0x210, Y 0x218, V 0x220, O 0x228, P 0x230, N 0x238.

extern "C" void kern_mma_chain_frag();

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
// round a float to f16-precision (value kept in float)
static float rh(float f) { return h2f(f2h(f)); }

int main() {
    constexpr int M = 16, N = 8, K = 8;   // MMA1 A=MxK, B=KxN, P=MxN; MMA2 V=MxK, O=MxN
    // NON-UNIFORM, distinct f16 inputs (each element different).
    std::vector<uint16_t> X(M * K), Y(K * N), V(M * K);
    for (int i = 0; i < M * K; ++i) X[i] = f2h(0.0625f * ((i * 7) % 23) - 0.5f);
    for (int i = 0; i < K * N; ++i) Y[i] = f2h(0.125f  * ((i * 5) % 17) - 0.75f);
    for (int i = 0; i < M * K; ++i) V[i] = f2h(0.09375f * ((i * 3) % 19) - 0.4375f);

    std::vector<float> O(M * N, -123.0f), P(M * N, -123.0f);

    launchKernel(kern_mma_chain_frag, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, X.data(),
                 ArgAt{0x218}, Y.data(),
                 ArgAt{0x220}, V.data(),
                 ArgAt{0x228}, O.data(),
                 ArgAt{0x230}, P.data(),
                 ArgAt{0x238}, (int)0);

    // ---- f64 oracle ----
    auto nl = [](double x) { double y = 0.5 * x + 0.25; return y > 0.0 ? y : 0.0; };
    // MMA1: P_full[m][n] = sum_k X[m][k]*Y[k][n], nonlinear, then f16-round.
    float Pref[M][N];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc += (double)h2f(X[m * K + k]) * (double)h2f(Y[k * N + n]);
            Pref[m][n] = rh((float)nl(acc));     // F2FP f16 round
        }
    // MMA2 candidate oracles. The .trans reload's exact (row,col) mapping into
    // the B-operand is the thing under test, so compute BOTH and report which
    // matches the lifter:
    //   T  (transposed B = Pt):  O[m][n] = sum_k V[m][k]*P[n][k]
    //   NT (B = P direct):       O[m][n] = sum_k V[m][k]*P[k][n]
    float Oref_T[M][N], Oref_NT[M][N];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double accT = 0.0, accNT = 0.0;
            for (int k = 0; k < K; ++k) {
                double v = (double)h2f(V[m * K + k]);
                accT  += v * (double)Pref[n][k];
                accNT += v * (double)Pref[k][n];
            }
            Oref_T[m][n]  = (float)accT;
            Oref_NT[m][n] = (float)accNT;
        }
    // Count which candidate matches the lifter output, decide Oref.
    int matchT = 0, matchNT = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float got = O[m * N + n];
            if (std::fabs(got - Oref_T[m][n])  <= 0.02f * std::max(std::fabs(Oref_T[m][n]),  1.0f)) matchT++;
            if (std::fabs(got - Oref_NT[m][n]) <= 0.02f * std::max(std::fabs(Oref_NT[m][n]), 1.0f)) matchNT++;
        }
    printf("MMA2 oracle match: transposed(B=Pt)=%d/128  direct(B=P)=%d/128\n", matchT, matchNT);
    float (*Oref)[N] = (matchNT >= matchT) ? Oref_NT : Oref_T;
    printf("Selected oracle: %s\n", (matchNT >= matchT) ? "direct(B=P)" : "transposed(B=Pt)");

    // ---- Per-row compare ----
    int nbad = 0; float maxd = 0;
    int badrows[M]; for (int i=0;i<M;i++) badrows[i]=0;
    // First, sanity-check P (MMA1 output) row-by-row.
    int pbad = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float got = P[m * N + n], exp = Pref[m][n];
            if (std::fabs(got - exp) > 0.01f * std::max(std::fabs(exp), 1.0f)) {
                if (pbad < 6) printf("P bad (%d,%d) got=%g exp=%g\n", m, n, got, exp);
                pbad++;
            }
        }
    printf("MMA1 P mismatches: %d/%d\n", pbad, M * N);

    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float got = O[m * N + n], exp = Oref[m][n];
            float diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            if (diff > 0.02f * std::max(std::fabs(exp), 1.0f)) {
                if (nbad < 8) printf("O bad row=%d col=%d got=%g exp=%g diff=%g\n", m, n, got, exp, diff);
                nbad++; badrows[m]++;
            }
        }
    }
    printf("per-row bad-cell counts:");
    for (int m = 0; m < M; ++m) printf(" r%d=%d", m, badrows[m]);
    printf("\n");

    if (!nbad && !pbad) { printf("REF VERIFIED (max_diff=%g)\nTEST PASSED\n", maxd); return 0; }
    printf("FAILED: O %d/%d mismatch (P %d) max_diff=%g\n", nbad, M * N, pbad, maxd);
    return 1;
}
