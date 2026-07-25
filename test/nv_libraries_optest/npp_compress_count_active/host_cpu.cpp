// host_cpu.cpp — x86 exec-correctness for a REAL NPP connected-components kernel.
//
// Kernel: CompressMarkerLabelsCountBlockLocalActiveIndexesUF<uint>(uint nTotal,
//             uint8_t* pBuffer)
//   _Z50CompressMarkerLabelsCountBlockLocalActiveIndexesUFjPh  (libnpp sm_90)
//
// Decoded from the full SASS (unrolled-by-4 loop + remainder loop + final store):
//   Survivor: only thread (tid.x==0 && tid.y==0 && ctaid.y==0); one lane per
//   block-x. Each block b = blockIdx.x owns the index range
//       lo = b*8192,  hi = min(lo + 8192, nTotal).
//   View pBuffer as uint*. Mark region is pBuffer[1..nTotal] (written by Pass1).
//   Per-block running active count (resets per block):
//       cnt = 0;
//       for (idx = lo; idx < hi; ++idx) {
//           v = pBuffer[idx + 1];                 // READ: (pBuffer+4) + idx*4
//           if (v != 0) { ++cnt; pBuffer[idx + nTotal + 2] = cnt; }
//           else        {        pBuffer[idx + nTotal + 2] = 0;   }
//       }                                          // WRITE: pBuffer + idx*4 + (nTotal+1)*4 + 4
//   Final per-block total:
//       pBuffer[2*nTotal + b + 2] = cnt;           // (pBuffer+4) + (2*nTotal+1+b)*4
//
// This is a block-local prefix count over the mark array: at each active position
// it stores the 1-based running count within the block; at inactive positions 0;
// and the block's total active count into a separate per-block slot. The output
// is a pure, deterministic function of the input mark array — each block writes a
// disjoint output region, so the wrapper's sequential CTAs reproduce it exactly.
//
// Discriminating input: a sparse mark array spanning 2 blocks (nTotal = 10000),
// with a non-trivial, position-dependent active pattern so a degenerate kernel
// (e.g. "write idx", "write 1 everywhere", or wrong region offset) fails.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void _Z50CompressMarkerLabelsCountBlockLocalActiveIndexesUFjPh();

static inline bool active(int idx) {
    // deterministic, irregular activity pattern (not all-on, not all-off)
    return ((idx * 2654435761u) >> 28) & 1u;   // ~50% active, scattered
}

int main() {
    constexpr uint32_t TILE = 8192;
    constexpr uint32_t nTotal = 10000;             // spans 2 blocks
    const int numBlocks = (int)((nTotal + TILE - 1) / TILE);   // 2

    // Buffer regions used: marks[1..nTotal], running[nTotal+2 .. 2*nTotal+1],
    // block-totals[2*nTotal+2 ..]. Allocate generously and zero-fill.
    const uint32_t BUFN = 3 * nTotal + 64;
    static uint32_t buf[3 * 10000 + 64];
    static uint32_t ref[3 * 10000 + 64];
    std::memset(buf, 0, sizeof(buf));

    // Lay down the mark array at pBuffer[idx+1].
    for (uint32_t idx = 0; idx < nTotal; ++idx)
        buf[idx + 1] = active((int)idx) ? (1u + (idx % 5)) : 0u;  // nonzero marks vary

    // poison the output regions so a "no-write" bug is caught
    for (uint32_t i = nTotal + 2; i < BUFN; ++i) buf[i] = 0xDEADBEEF;
    std::memcpy(ref, buf, sizeof(uint32_t) * BUFN);

    // Independent reference.
    for (int b = 0; b < numBlocks; ++b) {
        uint32_t lo = (uint32_t)b * TILE;
        uint32_t hi = lo + TILE; if (hi > nTotal) hi = nTotal;
        uint32_t cnt = 0;
        for (uint32_t idx = lo; idx < hi; ++idx) {
            uint32_t v = ref[idx + 1];
            if (v != 0) { ++cnt; ref[idx + nTotal + 2] = cnt; }
            else        {        ref[idx + nTotal + 2] = 0;   }
        }
        ref[2 * nTotal + (uint32_t)b + 2] = cnt;
    }

    launchKernel(_Z50CompressMarkerLabelsCountBlockLocalActiveIndexesUFjPh,
                 numBlocks, 32,                    // gridDim.x = numBlocks; 1 lane survives
                 ArgAt{0x210}, (uint32_t)nTotal,   // nTotal
                 ArgAt{0x218}, (uint32_t*)buf);    // pBuffer

    int bad = 0;
    for (uint32_t i = 0; i < BUFN; ++i)
        if (buf[i] != ref[i]) {
            if (bad < 8) printf("  buf[%u]: got %u want %u\n", i, buf[i], ref[i]);
            ++bad;
        }
    printf("=== Count: block totals = ");
    for (int b = 0; b < numBlocks; ++b) printf("%u ", buf[2 * nTotal + b + 2]);
    printf("(of nTotal=%u) ===\n", nTotal);
    printf("mismatches = %d / %u\n", bad, BUFN);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
