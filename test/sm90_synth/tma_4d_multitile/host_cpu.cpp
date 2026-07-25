// host_cpu.cpp for sm90_synth/tma_4d_multitile — x86 execution-correctness
// driver for the multi-tile UTMALDG.4D coordinate-recovery test.
//
// The kernel issues NTILE=4 genuine cp.async.bulk.tensor.4d loads, each with the
// SAME rank-4 CUtensorMap but a DIFFERENT coordinate vector {0, it*8, 0, 0}
// (the seq-tile row steps each iteration), and stores each loaded (8 x 64) tile
// straight back to gOut at its tile rows. So out rows [it*8, it*8+8) must equal
// input rows [it*8, it*8+8) — only correct if the lifter recovered each tile's
// distinct coordinate. With the OLD coord=0 behaviour every tile loaded rows
// [0,8) -> out rows 8..31 would be wrong.
//
// The CUtensorMap is opaque on x86, so this driver materializes a SlifterTmaDesc
// describing the whole (SEQ x DIM) input plane and points the kernel's
// descriptor load (c[0x0][0x198] + 0x30) at it. swizzle=4 = plain-linear (raw
// SWIZZLE_NONE box copy). The varying coordinate is supplied by the lifter's
// recovered coord operands (UR10..UR13), not by this driver.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
#include "embed.h"   // EMB_A[SEQ*DIM] u16  (head-0 input plane)

extern "C" void tma_4d_multitile_kernel();

static constexpr int DIM = 64, M_TILE = 8, NTILE = 4, SEQ = M_TILE * NTILE;

int main() {
    uint16_t hA[SEQ * DIM];
    uint16_t outBuf[SEQ * DIM];
    memcpy(hA, EMB_A, sizeof(hA));
    memset(outBuf, 0, sizeof(outBuf));

    // One descriptor for the whole (SEQ x DIM) plane; each UTMALDG.4D selects a
    // (M_TILE x DIM) tile via its recovered coords (m_base = c1 = it*M_TILE).
    SlifterTmaDesc descA{};
    descA.global_base   = hA;
    descA.rows          = M_TILE;   // rows per tile
    descA.cols_k        = DIM;      // full K (DIM) in ONE UTMALDG (no K-blocking)
    descA.elem_bytes    = 2;
    descA.swizzle       = 4;        // plain-linear (SWIZZLE_NONE box copy)
    descA.ld            = DIM;      // global row stride in elements
    descA.tile_base_off = 0;        // smem base offset (wrapper adds smem_off)

    // The kernel reads the descriptor POINTER at const[0x198] and dereferences
    // (pointer + 0x30) as the CUtensorMap. So const[0x198] = (&descA - 0x30).
    uint8_t* descA_as_seen = reinterpret_cast<uint8_t*>(&descA);
    uint64_t descA_param   = reinterpret_cast<uint64_t>(descA_as_seen) - 0x30;

    // The output STG.E uses desc[UR14] with UR14 = c[0x0][0x208]; that descriptor
    // base must be 0 so the store lands at gOut + offset unmodified. gOut itself
    // is c[0x0][0x2c0].
    launchKernel(tma_4d_multitile_kernel, 1, 128,
                 ArgAt{0x198}, (uint64_t)descA_param,
                 ArgAt{0x208}, (uint64_t)0,
                 ArgAt{0x2c0}, (uint16_t*)outBuf);

    auto h2f = [](uint16_t h) {
        uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
        uint32_t f;
        if (e == 0) { if (m == 0) f = s << 31; else { e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff;
            f = (s << 31) | (e << 23) | (m << 13); } }
        else if (e == 0x1f) f = (s << 31) | (0xff << 23) | (m << 13);
        else f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
        float r; memcpy(&r, &f, 4); return r;
    };

    int bad = 0, bad_tile1 = 0;
    for (int i = 0; i < SEQ * DIM; i++) if (outBuf[i] != hA[i]) bad++;
    // Specifically: tile 1 (rows 8..15) must equal input rows 8..15, NOT rows 0..7.
    for (int i = M_TILE * DIM; i < 2 * M_TILE * DIM; i++)
        if (outBuf[i] != hA[i]) bad_tile1++;

    printf("=== x86 4D multitile: tile0 row0 out[0..7] ===\n");
    for (int k = 0; k < 8; k++) printf("%.4f ", h2f(outBuf[k])); printf("\n");
    printf("=== tile1 (row 8) out, must == input row 8 ===\n");
    for (int k = 0; k < 8; k++) printf("%.4f ", h2f(outBuf[M_TILE * DIM + k])); printf("\n");
    printf("=== input row 8 ===\n");
    for (int k = 0; k < 8; k++) printf("%.4f ", h2f(hA[M_TILE * DIM + k])); printf("\n");
    printf("mismatches = %d / %d   (tile1 mismatches = %d)\n",
           bad, SEQ * DIM, bad_tile1);

    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL: %d / %d elements mismatch\n", bad, SEQ * DIM);
    return 1;
}
