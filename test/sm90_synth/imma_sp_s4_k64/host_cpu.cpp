// EXECUTING host driver for the int4 m16n8k128 2:4-sparse fixture (kernel.cu) — the
// VALUE-exactness companion to test/sm90_synth/imma_sp_s4 (whose host_cpu.cpp only
// DIRECT-drives the wrapper and never runs the lifted kernel).
//
// This host RUNS the LIFTED kernel via launchKernel (const-bank param injection) and
// compares the REAL lifted D to a dense int4 2:4 oracle. The single LOGICAL int4
// m16n8k128 PTX atom lowers to a PAIR of native K=64 IMMA.SP.16864 SASS sub-atoms whose
// D the kernel SUMS (atom2.D += atom1.D). The lifter routes EACH sub-atom to
// imma_sp_s4_k64_warp (the native K=64 int4 sparse op = the int8 imma_sp16864 K=64
// thread map reading int4 low-nibbles); the SASS sum then forms the K=128 result. This
// is the EXACT structure of CUTLASS 15_ampere_sparse_tensorop_gemm's warp atom.
//
// The kernel.cu reads:
//   Acomp[m*8 + (ccol>>3)] nibble (ccol&7)     ccol 0..63 (16 rows x 64 kept int4)
//   B[n*16 + (k>>3)]       nibble (k&7)         k    0..127 (8 rows x 128 int4)
//   C[m*8 + n] int32 ; D[m*8 + n] int32 ; Emeta[lane] uint32 (32 lanes)
// Param offsets (sm90, base 0x210): A 0x210, B 0x218, C 0x220, D 0x228, E 0x230.

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <array>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void
kern_imma_sp_s4_integer_subbyteILi4ELb1EE_SparseMmaINS6_ILi16ELi8ELi128EE();

static constexpr int M = 16, N = 8, K = 128, KC = 64;

// A-comp packed-nibble offset (lane t, nibble p in 0..31) — cute SM80_16x8x64 A.
static inline int aoff(int t, int p) {
    const int t0 = t & 3, t1 = t >> 2;
    return 128 * t0 + t1 + 16 * (p % 8) + 8 * ((p / 8) % 2) + 512 * ((p / 16) % 2);
}
// deterministic int4 2:4 PAIR selection (matches the shipped imma_sp_s4 host).
static inline void sel_of(int m, int dg, int& idx0, int& idx1) {
    const int pat = (m * 3 + dg * 5) & 0x3;
    switch (pat) {
        case 0: idx0 = 0; idx1 = 1; break;
        case 1: idx0 = 0; idx1 = 3; break;
        case 2: idx0 = 1; idx1 = 2; break;
        default: idx0 = 2; idx1 = 3; break;
    }
}

int main() {
    // dense oracle operands
    int32_t Adense[M][K]; int32_t Bdense[N][K];
    std::memset(Adense, 0, sizeof(Adense));
    std::memset(Bdense, 0, sizeof(Bdense));

    // kernel-layout buffers
    std::vector<uint32_t> Acomp(M * (KC / 8), 0);   // 16*8 = 128 uint32
    std::vector<uint32_t> Bk(N * (K / 8), 0);        // 8*16  = 128 uint32
    std::vector<int32_t>  C(M * N), D(M * N, -777);
    std::vector<uint32_t> Emeta(32, 0);

    auto setA = [&](int m, int ccol, int v) {        // ccol 0..63
        Acomp[m * 8 + (ccol >> 3)] |= (uint32_t(v & 0xF)) << (4 * (ccol & 7));
    };
    auto setB = [&](int n, int k, int v) {           // k 0..127
        Bk[n * 16 + (k >> 3)] |= (uint32_t(v & 0xF)) << (4 * (k & 7));
    };

    // Build A-comp + dense A + metadata E mirroring the imma_sp_s4 host enumeration
    // (the SAME enumeration the wrapper reads, so reconstruction is exact). HIGH-ENTROPY
    // int4 values spread the full [-8,7] so the .satfinite saturation path is exercised.
    for (int t = 0; t < 32; ++t) {
        for (int gg = 0; gg < 8; ++gg) {
            const int j0 = 2 * gg, p0 = 2 * j0;
            const int off = aoff(t, p0);
            const int m = off % 16, dg = (off / 16) / 4;
            int i0, i1; sel_of(m, dg, i0, i1);
            Emeta[t] |= (uint32_t((i0 & 3) | ((i1 & 3) << 2)) & 0xf) << (4 * gg);
            for (int sel = 0; sel < 2; ++sel) {
                const int j  = 2 * gg + sel;
                const int q0 = 2 * j, q1 = 2 * j + 1;
                const int offp = aoff(t, q0);
                const int mm = offp % 16, ccol = offp / 16;   // ccol 0..63
                const int idx = sel == 0 ? i0 : i1;
                const int v0 = (((mm + j * 5 + 1) & 0xF)) - 8;   // full int4 spread
                const int v1 = (((mm * 3 + j + 7) & 0xF)) - 8;
                setA(mm, ccol, v0);
                const int offp1 = aoff(t, q1);
                const int mm1 = offp1 % 16, ccol1 = offp1 / 16;
                setA(mm1, ccol1, v1);
                const int dg2 = ccol / 4;
                Adense[mm][8 * dg2 + 2 * idx]     = v0;
                Adense[mm][8 * dg2 + 2 * idx + 1] = v1;
            }
        }
        // B dense + packed (mirror wrapper read enumeration)
        const int t0 = t & 3, t1 = t >> 2;
        for (int p = 0; p < 32; ++p) {
            const int ps = p % 16;
            const int offB = 64 * t0 + t1 + 8 * (ps % 8) + 256 * (ps / 8);
            const int bn = offB % 8, bk = (offB / 8) + 64 * (p / 16);
            const int v = ((bn * 5 + bk * 3 + 2) % 7) - 3;
            setB(bn, bk, v);
            Bdense[bn][bk] = v;
        }
        // C
        for (int v = 0; v < 4; ++v) {
            const int m = t1 + 8 * (v / 2);
            const int n = t0 * 2 + (v % 2);
            C[m * N + n] = (m * N + n) % 13 - 6;
        }
    }

    // ---- RUN the lifted kernel ----
    launchKernel(kern_imma_sp_s4_integer_subbyteILi4ELb1EE_SparseMmaINS6_ILi16ELi8ELi128EE,
                 /*grid*/1, /*block*/32,
                 ArgAt{0x210}, Acomp.data(),
                 ArgAt{0x218}, Bk.data(),
                 ArgAt{0x220}, C.data(),
                 ArgAt{0x228}, D.data(),
                 ArgAt{0x230}, Emeta.data());

    // ---- oracle: dense int4 GEMM D = C + A_dense * B^T ----
    int nbad = 0; int64_t maxd = 0;
    int firstm = -1, firstn = -1; int64_t fg = 0, fe = 0;
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            int64_t acc = (m * N + n) % 13 - 6;   // C[m][n]
            for (int k = 0; k < K; ++k)
                acc += (int64_t)Adense[m][k] * (int64_t)Bdense[n][k];
            const int64_t diff = std::llabs((int64_t)D[m * N + n] - acc);
            maxd = std::max<int64_t>(maxd, diff);
            if (diff != 0) { if (!nbad) { firstm = m; firstn = n; fg = D[m * N + n]; fe = acc; } nbad++; }
        }
    }

    if (!nbad) {
        printf("REF VERIFIED (max_diff=%lld)\nTEST PASSED\n", (long long)maxd);
        return 0;
    }
    printf("FAILED: %d/%d mismatch max_diff=%lld; first (%d,%d) got=%lld exp=%lld\n",
           nbad, M * N, (long long)maxd, firstm, firstn, (long long)fg, (long long)fe);
    return 1;
}
