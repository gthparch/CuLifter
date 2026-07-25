#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the multi-tile (2x2) DMMA.8x8x4 f64 GEMM fixture.
//   D[i][j] = alpha * sum_{k<4} A[i][k]*B[j][k] + beta * C[i][j]   (16x16 output, 4 atoms)
//   A 16x4 row-major; B 16x4 (col-major K x N => B[j][k] is the operand element); C/D 16x16.
// This re-audits the C/D m<->n transpose across off-diagonal (mi != ni) output tiles.
//
// kern_gemm2x2(const double* A, const double* B, const double* C, double* D, double alpha, double beta)
// Param offsets (sm90, base 0x210): A 0x210, B 0x218, C 0x220, D 0x228, alpha 0x230, beta 0x238.

extern "C" void kern_gemm2x2();

int main() {
    constexpr int M = 16, N = 16, K = 4;
    const double CANARY = -55555.5;
    std::vector<double> A(M * K), B(N * K), C(M * N), D(M * N, CANARY);

    for (int i = 0; i < M * K; ++i) A[i] = 0.25 * ((i % 23) + 1) - 1.5;
    for (int i = 0; i < N * K; ++i) B[i] = 0.125 * ((i % 19) + 1) - 1.0;   // stored N x K
    for (int i = 0; i < M * N; ++i) C[i] = 0.5 * ((i % 13) + 1) - 2.0;
    const double alpha = 2.25, beta = 1.25;

    launchKernel(kern_gemm2x2, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data(),
                 ArgAt{0x230}, alpha,
                 ArgAt{0x238}, beta);

    int nbad = 0;
    double maxd = 0.0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k)
                acc = std::fma(A[i * K + k], B[j * K + k], acc);    // B is N x K
            const double exp = alpha * acc + beta * C[i * N + j];
            const double got = D[i * N + j];
            const double diff = std::fabs(got - exp);
            maxd = std::max(maxd, diff);
            if (diff > 1e-9 * std::max(std::fabs(exp), 1.0)) {
                if (!nbad)
                    printf("first bad (%d,%d) tile(%d,%d) got=%.17g exp=%.17g\n",
                           i, j, i / 8, j / 8, got, exp);
                nbad++;
            }
        }
    }
    printf("written=%d/%d  value-mismatches=%d  max_diff=%g\n", M * N, M * N, nbad, maxd);
    if (!nbad) {
        printf("REF VERIFIED (multi-tile 2x2 GEMM, max_diff=%g)\nTEST PASSED\n", maxd);
        return 0;
    }
    printf("FAILED: %d/%d mismatch max_diff=%g\n", nbad, M * N, maxd);
    return 1;
}
