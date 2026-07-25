#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the DMMA.8x8x4 f64 SYRK fixture (lower-fill, B := A => A·Aᵀ).
// One warp computes, for the lower triangle (i>=j) only:
//   D[i][j] = alpha * sum_k A[i][k]*A[j][k] + beta * C[i][j]
// and leaves the strict upper triangle (i<j) as a host-written canary.
//
// kern_syrk(const double* A, const double* C, double* D, double alpha, double beta)
// Param offsets (sm90, base 0x210): A 0x210, C 0x218, D 0x220, alpha 0x228, beta 0x230.

extern "C" void kern_syrk();

int main() {
    constexpr int M = 8, N = 8, K = 4;
    const double CANARY = -123456.5;        // distinctive, exact in f64
    std::vector<double> A(M * K), C(M * N), D(M * N, CANARY);

    // EXACT-in-f64 inputs (small q/8 rationals) so accumulation is the only rounding.
    for (int i = 0; i < M * K; ++i) A[i] = 0.25 * ((i % 23) + 1) - 1.5;
    for (int i = 0; i < M * N; ++i) C[i] = 0.5 * ((i % 13) + 1) - 2.0;
    const double alpha = 2.25, beta = 1.25;     // exact in f64

    launchKernel(kern_syrk, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, C.data(),
                 ArgAt{0x220}, D.data(),
                 ArgAt{0x228}, alpha,
                 ArgAt{0x230}, beta);

    int nbad = 0, ncanary_bad = 0, nwritten = 0;
    double maxd = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const double got = D[m * N + n];
            if (n <= m) {
                // lower triangle (incl diag): must equal alpha*A·Aᵀ + beta*C
                double acc = 0.0;
                for (int k = 0; k < K; ++k)
                    acc = std::fma(A[m * K + k], A[n * K + k], acc);
                const double exp = alpha * acc + beta * C[m * N + n];
                const double diff = std::fabs(got - exp);
                maxd = std::max(maxd, diff);
                nwritten++;
                if (diff > 1e-9 * std::max(std::fabs(exp), 1.0)) {
                    if (!nbad)
                        printf("first bad LOWER (%d,%d) got=%.17g exp=%.17g\n", m, n, got, exp);
                    nbad++;
                }
            } else {
                // strict upper triangle: must remain canary (unwritten)
                if (got != CANARY) {
                    if (!ncanary_bad)
                        printf("CANARY VIOLATION UPPER (%d,%d) got=%.17g (expected canary %.17g)\n",
                               m, n, got, CANARY);
                    ncanary_bad++;
                }
            }
        }
    }
    printf("lower-written=%d  upper-canary-violations=%d  value-mismatches=%d  max_diff=%g\n",
           nwritten, ncanary_bad, nbad, maxd);
    if (!nbad && !ncanary_bad) {
        printf("REF VERIFIED (SYRK A·Aᵀ lower-fill, max_diff=%g)\nTEST PASSED\n", maxd);
        return 0;
    }
    printf("FAILED: value-mismatch=%d canary-viol=%d max_diff=%g\n", nbad, ncanary_bad, maxd);
    return 1;
}
