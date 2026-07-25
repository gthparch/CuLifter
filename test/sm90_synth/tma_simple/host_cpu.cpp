// host_cpu.cpp for sm90_synth/tma_simple — x86 execution-correctness driver for
// a genuine sm_90a TMA bulk load (UTMALDG). The kernel TMA-loads a 64x8 K-INTER
// fp16 tile of A from global into shared memory, then stores it straight back to
// gOut. The CUtensorMap is opaque, so this driver materializes a SlifterTmaDesc
// of the same shape at the address the kernel's descriptor load resolves to
// (*(const[0x198]) + 0x30), and lift_utmaldg lowers UTMALDG to the wrapper's
// utma_load() which performs the real coord -> shared-memory tile copy.
//
// Verifies the round-tripped output equals the input (TMA moved the bytes).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
#include "embed.h"   // EMB_A[512] u16  (64x8 fp16 input, from the H200 run)

extern "C" void tma_simple_kernel();

int main() {
    constexpr int M = 64, K = 8;

    uint16_t hA[M * K];
    uint16_t outBuf[M * K];
    memcpy(hA, EMB_A, sizeof(hA));
    memset(outBuf, 0, sizeof(outBuf));

    // Materialize the TMA descriptor for A: linear K-major 64x8 fp16, no swizzle.
    SlifterTmaDesc descA{};
    descA.global_base   = hA;
    descA.rows          = M;
    descA.cols_k        = K;       // single K-block (K<=8)
    descA.elem_bytes    = 2;
    descA.swizzle       = 0;
    descA.ld            = K;
    descA.tile_base_off = 0x400;   // smem base this tile loads to (UPRMT result)

    // The kernel reads the descriptor POINTER at const[0x198] and dereferences
    // (pointer + 0x30) as the CUtensorMap. So const[0x198] must hold
    // (&descA - 0x30) -> (ptr + 0x30) == &descA.
    uint8_t* descA_as_seen = reinterpret_cast<uint8_t*>(&descA);
    uint64_t descA_param   = reinterpret_cast<uint64_t>(descA_as_seen) - 0x30;

    // Output store uses desc[UR6] with UR6 = const[0x208]; base must be 0 so the
    // store lands at gOut + i*2 unmodified.
    launchKernel(tma_simple_kernel, 1, 128,
                 ArgAt{0x198}, (uint64_t)descA_param,
                 ArgAt{0x208}, (uint64_t)0,
                 ArgAt{0x300}, (uint16_t*)outBuf);

    // Compare round-tripped output against the input (and the H200 reference,
    // which for this round-trip equals the input bit-for-bit).
    int bad = 0;
    for (int i = 0; i < M * K; i++) if (outBuf[i] != hA[i]) bad++;

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
    printf("=== x86 TMA round-trip out[0..7] ===\n");
    for (int k = 0; k < 8; k++) printf("%.4f ", h2f(outBuf[k])); printf("\n");
    printf("=== input            in[0..7]  ===\n");
    for (int k = 0; k < 8; k++) printf("%.4f ", h2f(hA[k])); printf("\n");
    printf("mismatches = %d / %d\n", bad, M * K);

    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL: %d / %d elements mismatch\n", bad, M * K);
    return 1;
}
