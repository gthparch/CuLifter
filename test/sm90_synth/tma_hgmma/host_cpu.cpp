// host_cpu.cpp for sm90_synth/tma_hgmma — x86 execution-correctness driver for a
// genuine sm_90a TMA->HGMMA kernel: TMA-loads A(64x16) and B(16x16) K-INTER fp16
// tiles global->shared (4 UTMALDG, 2 K-blocks each), then HGMMA.64x16x16.F32
// computes D[64x16] = A * B^T. This mirrors FlashAttention's operand load.
//
// The two CUtensorMaps are opaque and live in ONE contiguous block (the cute
// TiledCopy tuple) pointed to by const[0x198]:
//   descA at *(const[0x198]) + 0x30   (smem tile base 0x400)
//   descB at *(const[0x198]) + 0xf0   (smem tile base 0xC00)
// D output pointer is at const[0x3c0]. The HGMMA reads A/B back out of shared
// memory via the GMMA descriptor pair (already x86-verified in wgmma_ss_fa).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
#include "embed.h"   // EMB_A[1024] u16, EMB_B[256] u16, EMB_D[1024] f32

extern "C" void tma_hgmma_kernel();

int main() {
    constexpr int M = 64, K = 16, N = 16;

    uint16_t hA[M * K], hB[N * K];
    float    outD[M * N];
    memcpy(hA, EMB_A, sizeof(hA));
    memcpy(hB, EMB_B, sizeof(hB));
    memset(outD, 0, sizeof(outD));

    // Both descriptors live in one block; descA at base+0x30, descB at base+0xf0.
    alignas(64) static uint8_t descblk[0x200];
    SlifterTmaDesc* descA = reinterpret_cast<SlifterTmaDesc*>(descblk + 0x30);
    SlifterTmaDesc* descB = reinterpret_cast<SlifterTmaDesc*>(descblk + 0xf0);
    *descA = SlifterTmaDesc{};
    descA->global_base = hA; descA->rows = M; descA->cols_k = K;
    descA->elem_bytes = 2; descA->swizzle = 0; descA->ld = K;
    descA->tile_base_off = 0x400;
    *descB = SlifterTmaDesc{};
    descB->global_base = hB; descB->rows = N; descB->cols_k = K;
    descB->elem_bytes = 2; descB->swizzle = 0; descB->ld = K;
    descB->tile_base_off = 0xC00;

    uint64_t desc_param = reinterpret_cast<uint64_t>(descblk);

    launchKernel(tma_hgmma_kernel, 1, 128,
                 ArgAt{0x198}, (uint64_t)desc_param,
                 ArgAt{0x3c0}, (float*)outD);

    double maxabs = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs((double)outD[i] - (double)EMB_D[i]);
        if (diff > maxabs) maxabs = diff;
    }
    const double EPS_ABS = 1e-2;
    printf("=== x86 TMA->HGMMA D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", outD[n]); printf("\n");
    printf("=== native H200      D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", EMB_D[n]); printf("\n");
    printf("max |x86 - H200| = %.6f  (eps_abs=%.0e)\n", maxabs, EPS_ABS);

    int bad = 0;
    for (int i = 0; i < M * N; i++)
        if (fabs((double)outD[i] - (double)EMB_D[i]) > EPS_ABS) bad++;
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL: %d / %d elements exceed eps_abs\n", bad, M * N);
    return 1;
}
