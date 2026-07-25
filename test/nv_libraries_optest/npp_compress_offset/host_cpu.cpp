// host_cpu.cpp — x86 exec-correctness for a REAL NPP connected-components kernel.
//
// Kernel: CompressMarkerLabelsOffsetLocalBlockCountsByBlockSumBaseUF(uint a, uint b,
//             uint8_t* pBuffer)
//   _Z58CompressMarkerLabelsOffsetLocalBlockCountsByBlockSumBaseUFjjPh  (libnpp sm_90)
//
// Params (KPARAM_INFO): a @0x210 (uint), b @0x214 (uint), pBuffer @0x218 (uint8_t*).
// Survivor: only the lane with tid.x==tid.y==ctaid.y==0 (LOP3 P0 guard @0x40/0x50);
// gridDim.x picks the block.  View pBuffer as uint*.
//
// Decoded from full SASS (instr-by-instr):
//   For block g = ctaid.x:
//     lo  = g*8192
//     hi  = min(g*8192 + 8192, a)                 (VIADDMNMX @0x130)
//     if (lo >= hi) goto tail                      (@0x150/0x170)
//     Base = IN_u32[2*a + 2 + b*8192 + g]          (LDG @0x140, idx 2a+1+b*8192+g, +0x4)
//     running_max = 0
//     for (j = lo; j < hi; ++j) {                  (unrolled x4 + remainder)
//         idx = a + 2 + j                          (R17=a+1, +1 from +0x4)
//         v = IN_u32[idx]
//         if (v != 0) { v += Base; OUT_u32[idx] = v; running_max = max(running_max,v); }
//         // v==0 -> element left unchanged, not counted in max
//     }
//   tail (runs only if hi+8192 > a, i.e. the LAST block whose hi reaches a):
//     // @0x650..0x750
//     // R6 = hi+8192; if !(R6>a) EXIT.
//     Base2 = IN_u32[2*a + 2 + b*8192 + g]         (same address; == Base)
//     total = max(running_max, Base + Base2)       (VIADDMNMX @0x710, == max(rm, 2*Base))
//     if (total > IN_u32[0]) IN_u32[0] = total;    (@0x720/0x730/0x740)
//
// This is the "offset the per-block local counts by the running block-sum base, and
// publish the overall maximum label into slot 0" step of the label-compaction pipeline.
// Deterministic, order-independent in v (one lane scans the block serially).
//
// Discriminating test: ONE block (gridDim.x=1 -> g=0, lo=0, hi=min(8192,a)=a since
// a<8192). Tail FIRES (hi+8192 = a+8192 > a).  Base is a chosen nonzero offset; the
// IN region has a mix of zeros (must stay untouched) and nonzeros (must get +Base and
// feed the max).  Slot 0 gets max(running_max, 2*Base).  A degenerate kernel
// (copy / unconditional add / no max) mismatches.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void _Z58CompressMarkerLabelsOffsetLocalBlockCountsByBlockSumBaseUFjjPh();

int main() {
    constexpr uint32_t A = 100;     // a: lo=0, hi=min(8192,100)=100 -> scan idx [a+2 .. a+2+99]
    constexpr uint32_t B = 0;       // b: base-region offset b*8192 = 0 (keeps layout compact)
    constexpr uint32_t G = 1;       // gridDim.x = 1 block -> g=0

    const uint32_t baseIdx = 2*A + 2 + B*8192 + 0; // = 202 : per-block Base slot
    const uint32_t inLo    = A + 2;                // = 102 : first scanned element
    const uint32_t inHi    = A + 2 + A;            // = 202 : one past last (hi=a=100)
    // note: inHi (202) == baseIdx (202). The Base slot sits right after the IN window,
    // and is itself NOT scanned (j<hi excludes idx=a+2+hi). Good — disjoint.

    const uint32_t BUFN = baseIdx + 64;
    static uint32_t buf[2*100 + 2 + 64 + 8];
    static uint32_t ref[2*100 + 2 + 64 + 8];
    std::memset(buf, 0, sizeof(buf));

    const uint32_t BASE = 1000;     // the per-block running-sum base offset to add
    buf[baseIdx] = BASE;

    // IN window [inLo, inHi): irregular, ~1/4 zeros (must be left untouched).
    for (uint32_t j = 0; j < A; ++j) {
        uint32_t k = inLo + j;
        uint32_t v = ((k * 2654435761u) >> 26) % 16;  // 0..15, ~1/16 are 0
        buf[k] = v;
    }
    // slot 0 starts below what the tail will write, so the max-write is observable.
    buf[0] = 5;

    std::memcpy(ref, buf, sizeof(uint32_t) * BUFN);

    // Independent reference.
    uint32_t Base = ref[baseIdx];
    uint32_t running_max = 0;
    for (uint32_t j = 0; j < A; ++j) {       // lo=0..hi=a (single block)
        uint32_t idx = A + 2 + j;
        uint32_t v = ref[idx];
        if (v != 0) {
            v += Base;
            ref[idx] = v;
            if (v > running_max) running_max = v;
        }
    }
    // tail: hi+8192 = a+8192 > a -> fires.
    uint32_t total = running_max;
    uint32_t twoBase = Base + Base;     // R6(Base reload) + R4(Base) -> VIADDMNMX max
    if (twoBase > total) total = twoBase;
    if (total > ref[0]) ref[0] = total;

    launchKernel(_Z58CompressMarkerLabelsOffsetLocalBlockCountsByBlockSumBaseUFjjPh,
                 G, 32,                          // G blocks, 1 lane survives each
                 ArgAt{0x210}, (uint32_t)A,      // a
                 ArgAt{0x214}, (uint32_t)B,      // b
                 ArgAt{0x218}, (uint32_t*)buf);  // pBuffer

    int bad = 0;
    for (uint32_t i = 0; i < BUFN; ++i)
        if (buf[i] != ref[i]) {
            if (bad < 8) printf("  buf[%u]: got %u want %u\n", i, buf[i], ref[i]);
            ++bad;
        }
    printf("=== Offset: Base=%u, slot0 got=%u ref=%u, sample IN+Base: ===\n",
           BASE, buf[0], ref[0]);
    printf("  buf[%u..%u] = ", inLo, inLo+7);
    for (uint32_t j = 0; j < 8; ++j) printf("%u ", buf[inLo+j]);
    printf("\n");
    printf("mismatches = %d / %u\n", bad, BUFN);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
