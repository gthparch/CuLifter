#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the DMMA.16x8x16 f64 fixture. One warp computes
// D = C + A(16x16) * B(16x8); the host replays the same matmul with the
// same ascending-k fma chain and compares.
//
// Param offsets (sm90, base 0x210): 0x210 A, 0x218 B, 0x220 C, 0x228 D.

extern "C" void kern_dmma16816();

int main() {
    constexpr int M = 16, N = 8, K = 16;
    std::vector<double> A(M * K), B(K * N), C(M * N), D(M * N, -7.0);
    for (int i = 0; i < M * K; ++i) A[i] = 0.25 * ((i % 23) + 1) - 1.5;
    for (int i = 0; i < K * N; ++i) B[i] = 0.125 * ((i % 19) + 1) - 1.0;
    for (int i = 0; i < M * N; ++i) C[i] = 0.5 * ((i % 13) + 1);

    launchKernel(kern_dmma16816, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data());

    int nbad = 0;
    double maxd = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            double acc = C[m * N + n];
            for (int k = 0; k < K; ++k)
                acc = std::fma(A[m * K + k], B[k * N + n], acc);
            const double diff = std::fabs(D[m * N + n] - acc);
            maxd = std::max(maxd, diff);
            if (diff > 1e-9 * std::max(std::fabs(acc), 1.0)) {
                if (!nbad)
                    printf("first bad (%d,%d) got=%g exp=%g\n",
                           m, n, D[m * N + n], acc);
                nbad++;
            }
        }
    }
    if (!nbad) {
        printf("REF VERIFIED (max_diff=%g)\nTEST PASSED\n", maxd);
        return 0;
    }
    printf("FAILED: %d/%d mismatch max_diff=%g\n", nbad, M * N, maxd);
    return 1;
}
