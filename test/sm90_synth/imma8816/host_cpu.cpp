#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the IMMA.8816 int8 tensor-core fixture. One warp computes the
// m8n8k16 atom
//   D[m][n] = C[m][n] + sum_k A[m][k]*B[n][k]   (M=N=8, K=16, int8 A/B, int32 C/D)
// with A (8x16 row-major s8), B stored (8x16)=(N x K) i.e. column-major K x N s8,
// C/D (8x8 row-major int32). Integer GEMM is EXACT (no rounding), so the compare
// is bit-exact.
//
// Param offsets (sm90, base 0x210): 0x210 A, 0x218 B, 0x220 C, 0x228 D.

extern "C" void kern_imma8816();

int main() {
    constexpr int M = 8, N = 8, K = 16;
    std::vector<signed char> A(M * K), B(N * K);
    std::vector<int32_t> C(M * N), D(M * N, -777);
    // Small int8 values so accumulation stays well within int32 (and is exact).
    for (int i = 0; i < M * K; ++i) A[i] = static_cast<signed char>((i % 11) - 5);
    for (int i = 0; i < N * K; ++i) B[i] = static_cast<signed char>((i % 7)  - 3);
    for (int i = 0; i < M * N; ++i) C[i] = (i % 13) - 6;

    launchKernel(kern_imma8816, /*grid*/1, /*block*/32,
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
