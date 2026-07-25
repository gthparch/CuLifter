#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for symm_dual_mma: D = A1*B + A2*B + C, two HMMA.1688.F32.TF32 into
// one warp fragment, each A staged in smem + LDSM.16.M88.4. tf32-snapped A/B
// => products exact in f32; per-(m,n) reference in f64. Reports per-ROW status
// to expose the SYMM (g&2)-bimodal lane bug. Param offsets (base 0x210):
//   A1 0x210, A2 0x218, B 0x220, C 0x228, D 0x230.

extern "C" void kern_dual_hmma_frag();

static float snap_tf32(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    x &= 0xFFFFE000u;
    float r; std::memcpy(&r, &x, 4); return r;
}

int main() {
    constexpr int M = 16, N = 8, K = 8;
    std::vector<float> A1(M*K), A2(M*K), B(K*N), C(M*N), D(M*N, -7.0f);
    uint32_t s = 0x5377F32u;
    auto nxt = [&]() { s = s * 1664525u + 1013904223u; return s; };
    auto rnd = [&](float sc) { return sc * (((int)(nxt() >> 16 & 0x3FF) - 512) / 256.0f); };
    for (int i = 0; i < M*K; ++i) A1[i] = snap_tf32(rnd(1.0f));
    for (int i = 0; i < M*K; ++i) A2[i] = snap_tf32(rnd(1.0f));
    for (int i = 0; i < K*N; ++i) B[i]  = snap_tf32(rnd(1.0f));
    for (int i = 0; i < M*N; ++i) C[i]  = rnd(2.0f);

    launchKernel(kern_dual_hmma_frag, /*grid*/1, /*block*/32,
                 ArgAt{0x210}, A1.data(),
                 ArgAt{0x218}, A2.data(),
                 ArgAt{0x220}, B.data(),
                 ArgAt{0x228}, C.data(),
                 ArgAt{0x230}, D.data());

    int nbadrows = 0, nbad = 0; double maxd = 0;
    printf("per-row status (g = row in 0..7 within each 8-row half):\n");
    for (int m = 0; m < M; ++m) {
        int rowbad = 0; double rowmax = 0;
        for (int n = 0; n < N; ++n) {
            double acc = C[m*N + n];
            for (int k = 0; k < K; ++k)
                acc += (double)A1[m*K + k] * (double)B[k*N + n]
                     + (double)A2[m*K + k] * (double)B[k*N + n];
            double got = D[m*N + n];
            double diff = std::fabs(got - acc);
            double rel = diff / std::max(std::fabs(acc), 1.0);
            rowmax = std::max(rowmax, rel);
            maxd = std::max(maxd, rel);
            if (rel > 1e-3) { rowbad++; nbad++; }
        }
        int gg = m % 8;
        printf("  row %2d (g=%d, g&2=%d): %s  maxrel=%.4g\n",
               m, gg, (gg & 2) ? 1 : 0, rowbad ? "BAD " : "GOOD", rowmax);
        if (rowbad) nbadrows++;
    }
    if (!nbad) { printf("REF VERIFIED (max_rel=%g)\nTEST PASSED\n", maxd); return 0; }
    printf("FAILED: %d/%d elems, %d/%d rows bad, max_rel=%g\n", nbad, M*N, nbadrows, M, maxd);
    return 1;
}
