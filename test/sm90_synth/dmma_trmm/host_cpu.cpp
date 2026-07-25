#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the DMMA.8x8x4 f64 TRMM fixture (left-side, lower-fill, non-unit diag, K=8).
// One warp computes the FULL output:
//   D[i][j] = alpha * sum_{k=0..7} A[i][k]*B[k][j]      (A lower-triangular: A[i][k]=0 for k>i)
// via TWO accumulated m8n8k4 atoms (K=8 = 2x k4).
//
// kern_trmm(const double* A, const double* Bcm, double* D, double alpha)
//   A   = 8x8 row-major, LOWER-triangular (strict upper zeroed).
//   Bcm = 8x8 stored col-major: Bcm[n*8 + k] = B[k][n].
// Param offsets (sm90, base 0x210): A 0x210, Bcm 0x218, D 0x220, alpha 0x228.

extern "C" void kern_trmm();

int main() {
    constexpr int M = 8, N = 8, K = 8;
    const double CANARY = -98765.25;
    std::vector<double> A(M * K, 0.0), B(K * N), Bcm(N * K), D(M * N, CANARY);

    // EXACT-in-f64 inputs. A is LOWER-triangular: only A[i][k] for k<=i is nonzero.
    for (int i = 0; i < M; ++i)
        for (int k = 0; k <= i; ++k)
            A[i * K + k] = 0.25 * (((i * 8 + k) % 23) + 1) - 1.5;
    // B (logical K x N) and its column-major storage Bcm[n][k] = B[k][n].
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            const double bv = 0.125 * (((k * 8 + n) % 19) + 1) - 1.0;
            B[k * N + n]  = bv;
            Bcm[n * K + k] = bv;
        }
    const double alpha = 1.75;       // exact in f64

    launchKernel(kern_trmm, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, Bcm.data(),
                 ArgAt{0x220}, D.data(),
                 ArgAt{0x228}, alpha);

    int nbad = 0;
    double maxd = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc = std::fma(A[i * K + k], B[k * N + j], acc);   // A lower-tri -> k>i terms are 0
            const double exp = alpha * acc;
            const double got = D[i * N + j];
            const double diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            if (diff > 1e-9 * std::max(std::fabs(exp), 1.0)) {
                if (!nbad)
                    printf("first bad (%d,%d) got=%.17g exp=%.17g\n", i, j, got, exp);
                nbad++;
            }
        }
    }
    printf("written=%d/%d  value-mismatches=%d  max_diff=%g\n", M * N, M * N, nbad, maxd);
    if (!nbad) {
        printf("REF VERIFIED (TRMM lower-tri left-side, max_diff=%g)\nTEST PASSED\n", maxd);
        return 0;
    }
    printf("FAILED: %d/%d mismatch max_diff=%g\n", nbad, M * N, maxd);
    return 1;
}
