#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the IMMA.SP.16864 int8 2:4-sparse tensor-core fixture. One warp
// computes the m16n8k64 sparse atom
//   D[m][n] = C[m][n] + sum_k A_dense[m][k]*B[n][k]   (M=16, N=8, K=64)
// where A is given COMPRESSED (M16 x 32 = 32 kept of 64 K-cols/row) plus metadata
// E selecting, per group of 4 K-cols, which 2 dense positions the kept values
// occupy (CUTLASS host_uncompress.h 2:4 rule). B is dense N8 x K64. Integer GEMM
// is EXACT, so the compare is bit-exact.
//
// We pick a deterministic sparsity pattern sel(row,g)->(idx0,idx1), build the
// compressed A by gathering A_dense's kept columns, and build the per-lane uint32
// metadata E in the layout kern_imma_sp16864 reads (and imma_sp16864_s8_warp
// decodes). The host oracle below reconstructs A_dense the same way.
//
// Param offsets (sm90, base 0x210): 0x210 A(comp), 0x218 B, 0x220 C, 0x228 D,
//                                   0x230 E.

extern "C" void kern_imma_sp16864();

// ---- shared geometry (must match kernel.cu + the warp wrapper) -------------
static inline int alayout_off(int t, int p) {
    const int t0 = t & 3, t1 = t >> 2;
    return 64 * t0 + t1 + 16 * (p % 4) + 8 * ((p / 4) % 2) + 256 * ((p / 8) % 2);
}
// deterministic 2:4 selection: for dense group g of row m, the two NON-zero dense
// positions (idx0<idx1) within [0..3]. Cover a spread of patterns over (m,g).
static inline void sel_of(int m, int g, int& idx0, int& idx1) {
    const int pat = (m * 3 + g * 5) & 0x3;   // 0..3 -> one of the 6 (i0,i1) pairs
    switch (pat) {
        case 0: idx0 = 0; idx1 = 1; break;
        case 1: idx0 = 0; idx1 = 3; break;
        case 2: idx0 = 1; idx1 = 2; break;
        default: idx0 = 2; idx1 = 3; break;
    }
}

int main() {
    constexpr int M = 16, N = 8, K = 64, KC = 32;   // KC = K/2 compressed cols
    std::vector<signed char> Acomp(M * KC), B(N * K);
    std::vector<int32_t> C(M * N), D(M * N, -777);
    std::vector<uint32_t> Emeta(32, 0);

    // dense A we are emulating: A_dense[m][k]. Only the 2:4-selected cols are
    // non-zero; the kept values are packed into Acomp[m][2g+{0,1}].
    std::vector<int32_t> Adense(M * K, 0);
    for (int m = 0; m < M; ++m) {
        for (int g = 0; g < 16; ++g) {           // 16 groups of 4 over K=64
            int i0, i1; sel_of(m, g, i0, i1);
            // kept values (small s8 so accumulation is exact)
            const int v0 = ((m * 7 + g * 3 + 1) % 11) - 5;
            const int v1 = ((m * 5 + g * 2 + 4) % 9) - 4;
            Adense[m * K + 4 * g + i0] = v0;
            Adense[m * K + 4 * g + i1] = v1;
            Acomp[m * KC + 2 * g + 0] = static_cast<signed char>(v0);
            Acomp[m * KC + 2 * g + 1] = static_cast<signed char>(v1);
        }
    }
    for (int i = 0; i < N * K; ++i) B[i] = static_cast<signed char>((i % 7) - 3);
    for (int i = 0; i < M * N; ++i) C[i] = (i % 13) - 6;

    // build per-lane metadata E.  CANONICAL pair-indexed scheme (identical on the
    // host, kernel.cu, and the warp wrapper): a lane's 16 compressed s8 fragments
    // form 8 PAIRS p=0,2,4,..,14 (pair j = p/2).  Pair j = compressed cols {2j-ish}
    // for one (row m, group g) via alayout_off; place that group's 4-bit selector
    // (i0 | i1<<2) at slot j (bits [4j .. 4j+3]) of E.  The wrapper reads slot j
    // and reconstructs the SAME (m, g) from the SAME alayout enumeration, so the
    // layout is self-consistent by construction.
    for (int t = 0; t < 32; ++t) {
        uint32_t e = 0;
        for (int j = 0; j < 8; ++j) {
            const int p   = 2 * j;
            const int off = alayout_off(t, p);
            const int m   = off % 16;
            const int g   = (off / 16) / 2;          // ccol/2 = dense group index
            int i0, i1; sel_of(m, g, i0, i1);
            e |= (static_cast<uint32_t>((i0 & 0x3) | ((i1 & 0x3) << 2)) & 0xf) << (4 * j);
        }
        Emeta[t] = e;
    }

    launchKernel(kern_imma_sp16864, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, Acomp.data(),
                 ArgAt{0x218}, B.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data(),
                 ArgAt{0x230}, Emeta.data());

    int nbad = 0;
    int64_t maxd = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = C[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += static_cast<int64_t>(Adense[m * K + k]) *
                       static_cast<int64_t>(B[n * K + k]);
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
