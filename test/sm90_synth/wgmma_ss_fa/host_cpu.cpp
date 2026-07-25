// host_cpu.cpp for sm90_synth/wgmma_ss_fa — x86 execution-correctness driver
// for a genuine sm_90a HGMMA (SS-form shared-memory descriptor) tensor-core
// kernel: D[64x16] = A[64x16] * B[16x16]^T, fp16 in / fp32 accum.
//
// The cubin is a REAL sm_90a build (HGMMA.64x16x16.F32 present, 0 TRAP) emitted
// by cute (CUTLASS 3.5). The lifted kernel reads gA/gB from global memory via
// desc[UR8] addressing, stages them into shared memory (STS), runs the HGMMA —
// which the x86 lifter decodes into a genuine A*B product by reading the A and B
// fragments back out of shared memory using the GMMA descriptor pair — and
// stores D to gD. Each of the 128 warpgroup threads owns a distinct D fragment
// (cute CLayout_64x16), computed from its SR_TID.X at runtime.
//
// Inputs (EMB_A/EMB_B) and the native ground truth (EMB_D) are embedded from a
// verified run on an NVIDIA GH200 (H200, sm_90a; max |HGMMA - cpu_fp32| =
// 0.00000). Scalar-decomposed HGMMA is not bit-exact vs the HW tensor core, so
// the assertion uses a documented absolute tolerance.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
#include "embed.h"   // EMB_A[1024] u16, EMB_B[256] u16, EMB_D[1024] f32

extern "C" void fa_wgmma();

int main() {
    constexpr int M = 64, K = 16, N = 16;

    // Mutable copies the lifted kernel reads (LDG.E.U16 reads these bytes).
    uint16_t hA[M * K], hB[N * K];
    float    outD[M * N];
    memcpy(hA, EMB_A, sizeof(hA));
    memcpy(hB, EMB_B, sizeof(hB));
    memset(outD, 0, sizeof(outD));

    launchKernel(fa_wgmma, 1, 128,
                 ArgAt{0x210}, (uint16_t*)hA,
                 ArgAt{0x218}, (uint16_t*)hB,
                 ArgAt{0x220}, (float*)outD);

    double maxabs = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs((double)outD[i] - (double)EMB_D[i]);
        if (diff > maxabs) maxabs = diff;
    }
    const double EPS_ABS = 1e-2;  // values are O(0.1); fp32-FMA vs HW tensor core

    printf("=== x86-lifted HGMMA D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", outD[n]); printf("\n");
    printf("=== native H200   D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", EMB_D[n]); printf("\n");
    printf("max |x86 - H200| = %.6f  (eps_abs=%.0e)\n", maxabs, EPS_ABS);

    int bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabs((double)outD[i] - (double)EMB_D[i]) > EPS_ABS) bad++;

    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL: %d / %d elements exceed eps_abs\n", bad, M * N);
    return 1;
}
