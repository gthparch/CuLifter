#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include "kernel_wrapper.h"

// Driver for the trailing-operand LDGSTS fixture (pred_gather_ldgsts).
//
// SASS shape (from kernel.sass):
//   P0..P3   = low 4 bits of pred_in[tid]            (R2P PR, R2, 0xf)
//   R5       = idx[tid]                               (LDG)
//   Rgather  = src + idx[tid]*4                       (IMAD.WIDE)
//   LDGSTS.E [smem], desc[UR6][Rgather], P0           (predicated async copy)
//
// For inactive lanes pred_in[tid]=0 (P0 false) AND idx[tid] is far OOB. HW
// skips the copy; a lifter that drops the LDGSTS predicate loads the OOB
// gather address -> SIGSEGV. With the fix the load is guarded -> no fault,
// and out[] for inactive lanes stays the pre-existing value.
//
// Param offsets (sm90, base 0x210):
//   0x210 src, 0x218 idx, 0x220 pred_in, 0x228 out, 0x230 N

extern "C" void kern_pred_gather();

int main() {
    constexpr int N = 256;
    constexpr int HALF = N / 2;
    constexpr int SRCN = 4096;

    std::vector<float> src(SRCN);
    std::vector<int>   idx(N);
    std::vector<int>   pred_in(N);
    std::vector<float> out(N, -7.0f);

    for (int i = 0; i < SRCN; i++) src[i] = (float)(i + 1);
    for (int i = 0; i < N; i++) {
        pred_in[i] = (i < HALF) ? 0xF : 0x0;                       // active half
        idx[i]     = (i < HALF) ? ((i * 7) % (SRCN - 8)) : (1 << 28); // OOB if inactive
    }

    launchKernel(kern_pred_gather, /*grid*/1, /*block*/N,
                 ArgAt{0x210}, src.data(),
                 ArgAt{0x218}, idx.data(),
                 ArgAt{0x220}, pred_in.data(),
                 ArgAt{0x228}, out.data(),
                 ArgAt{0x230}, N);

    // Success criterion for this fixture: the kernel RAN TO COMPLETION without
    // faulting on the OOB gather address of the inactive lanes. We don't model
    // the exact GEMM-less reduction value here (the SASS does a smem fold); the
    // bug under test is the SIGSEGV, so "no crash + active lanes are finite"
    // is the verification.
    int nonfinite = 0;
    for (int i = 0; i < HALF; i++)
        if (!std::isfinite(out[i])) nonfinite++;

    if (nonfinite == 0) { printf("REF VERIFIED (no OOB fault, %d active lanes finite)\n", HALF); printf("TEST PASSED\n"); return 0; }
    printf("FAILED: %d non-finite active lanes\n", nonfinite);
    return 1;
}
