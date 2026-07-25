// Host driver for the shflsubst fixture: 8 warps (one per column) solve
// L x = alpha*b against a double-precision host replica.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void kern_shflsubst();

int main()
{
    constexpr int N = 32, NCOL = 8;
    static float A[N * N];      // column-major, lower triangle used
    static float b[NCOL * N];
    static float x[NCOL * N];

    // Well-conditioned L: dominant diagonal, modest off-diagonal terms.
    for (int c = 0; c < N; ++c)
        for (int r = 0; r < N; ++r)
            A[c * N + r] = (r == c) ? 1.5f + 0.03125f * (float)((r * 5) % 11)
                                    : 0.0625f * (float)((r * 7 + c * 3) % 13 + 1);
    for (int i = 0; i < NCOL * N; ++i)
        b[i] = 0.1f * (float)((i % 17) + 1);
    std::memset(x, 0, sizeof(x));

    const float alpha = 2.5f;

    // Params (sm90 base 0x210): A 0x210, b 0x218, x 0x220, alpha 0x228.
    launchKernel2D(kern_shflsubst, Dim3{1, 1, 1}, Dim3{N, NCOL, 1},
                   ArgAt{0x210}, (const float*)A,
                   (const float*)b, (float*)x, alpha);

    // Double-precision replica of the substitution.
    int n_fail = 0;
    double max_rel = 0.0;
    for (int col = 0; col < NCOL; ++col) {
        double xs[N];
        for (int r = 0; r < N; ++r) {
            double acc = 0.0;
            for (int k = 0; k < r; ++k)
                acc += (double)A[k * N + r] * xs[k];
            xs[r] = ((double)alpha * (double)b[col * N + r] - acc)
                    / (double)A[r * N + r];
        }
        for (int r = 0; r < N; ++r) {
            const double got = (double)x[col * N + r];
            const double ref = xs[r];
            const double rel = std::fabs(got - ref)
                               / std::max(std::fabs(ref), 1.0);
            if (rel > max_rel) max_rel = rel;
            if (rel > 1e-3) {
                if (n_fail < 6)
                    std::printf("MISMATCH col=%d row=%d got=%.8g ref=%.8g\n",
                                col, r, got, ref);
                ++n_fail;
            }
        }
    }
    if (n_fail == 0) {
        std::printf("TEST PASSED (max_rel=%.3g)\n", max_rel);
        return 0;
    }
    std::printf("FAIL: %d/%d mismatches (max_rel=%.3g)\n",
                n_fail, NCOL * N, max_rel);
    return 1;
}
