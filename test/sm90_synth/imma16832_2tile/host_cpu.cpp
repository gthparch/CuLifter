#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the IMMA.16832 MULTI-TILE int8 GEMM fixture. One warp computes
//   D[m][n] = C[m][n] + sum_k A[m][k]*B[n][k]   (M=32, N=16, K=64, int8 A/B, int32 C/D)
// as 2x2 output tiles (M in tiles of 16, N in tiles of 8) each accumulating 2
// IMMA.16832 atoms over K (=64). Integer GEMM is EXACT. The off-diagonal output
// tiles make this catch any C/D m<->n transpose the single-atom imma16832 fixture
// cannot.
//
// Param offsets (sm90, base 0x210): 0x210 A, 0x218 B, 0x220 C, 0x228 D.

extern "C" void kern_imma16832_2tile();

int main() {
    constexpr int M = 32, N = 16, K = 64;
    std::vector<signed char> A(M * K), B(N * K);
    std::vector<int32_t> C(M * N), D(M * N, -777);
    // Asymmetric fills so a transpose would NOT accidentally match.
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<signed char>(((i * 3 + 1) % 11) - 5);
    for (int i = 0; i < N * K; ++i) B[i] = static_cast<signed char>(((i * 5 + 2) % 7)  - 3);
    for (int i = 0; i < M * N; ++i) C[i] = ((i * 7) % 13) - 6;

    launchKernel(kern_imma16832_2tile, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data());

    int nbad = 0;
    int64_t maxd = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = C[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += static_cast<int64_t>(A[m * K + k]) *
                       static_cast<int64_t>(B[n * K + k]);   // B is N x K
            const int64_t diff = std::llabs(static_cast<int64_t>(D[m * N + n]) - acc);
            maxd = std::max<int64_t>(maxd, diff);
            if (diff != 0) {
                if (!nbad)
                    printf("first bad (%d,%d) got=%d exp=%lld\n",
                           m, n, D[m * N + n], static_cast<long long>(acc));
                nbad++;
            }
        }
    }
    if (!nbad) {
        printf("REF VERIFIED (max_diff=%lld)\nTEST PASSED\n",
               static_cast<long long>(maxd));
        return 0;
    }
    printf("FAILED: %d/%d mismatch max_diff=%lld\n", nbad, M * N,
           static_cast<long long>(maxd));
    return 1;
}
