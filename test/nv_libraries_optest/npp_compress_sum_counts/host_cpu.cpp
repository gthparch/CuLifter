// host_cpu.cpp — x86 exec-correctness for a REAL NPP connected-components kernel.
//
// Kernel: CompressMarkerLabelsSumLocalBlockCountsUF<uint>(uint a, uint b, uint c,
//             uint8_t* pBuffer)
//   _Z41CompressMarkerLabelsSumLocalBlockCountsUFjjjPh  (libnpp sm_90)
//
// Decoded from the full SASS (carry-in step + unrolled main loop + remainder):
//   Survivor: only thread (tid.x==tid.y==ctaid.x==ctaid.y==0); a single lane.
//   Params: a @0x210, b @0x214, c @0x218, pBuffer @0x220.
//   lo = (a*c) * 8192;   hi = min(lo + 8192, b);   if (lo >= hi) return;
//   View pBuffer as uint*. Two regions, offset by base = 2*b + 2:
//       input  region: IN[idx]  = pBuffer[idx + 2*b + 2]
//       output region: OUT[idx] = pBuffer[idx + c*8192 + 2*b + 2]
//   The kernel computes an EXCLUSIVE prefix sum of IN into OUT:
//       if (lo == 0)  OUT[0] = 0;                 // seed (store RZ at base)
//       else          OUT[lo] = OUT[lo-1];        // carry-in from previous block
//       for (j = (lo==0?1:lo); j <= hi; ++j)
//           OUT[j] = OUT[j-1] + IN[j-1];          // running prefix sum
//   i.e. OUT[j] = sum_{k=0}^{j-1} IN[k]. (Recurrence verified instr-by-instr:
//   LDG OUT[j-1], LDG IN[j-1], IMAD.IADD, STG OUT[j].)
//
// This is the standard "sum the per-block local counts into running offsets" scan
// of the label-compaction pipeline. Output is a deterministic, order-independent
// function of IN (serial scan by one lane) — matches the wrapper exactly.
//
// We exercise a SINGLE block (a=0 -> lo=0; b<=8192 -> hi=b) so the whole array is
// scanned in one pass over both the unrolled and remainder loops, and we choose
// c=1 so the OUT region (idx + 8192 + ...) is disjoint from the IN region.
// Discriminating input: an irregular IN[] so the running prefix sum is strictly
// increasing with varying steps (a degenerate kernel — copy/constant/off-by-one —
// fails).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void _Z41CompressMarkerLabelsSumLocalBlockCountsUFjjjPh();

int main() {
    constexpr uint32_t A = 0;        // lo = a*c*8192 = 0  -> single block, lo==0 seed path
    constexpr uint32_t C = 1;        // OUT region offset by c*8192 (disjoint from IN)
    constexpr uint32_t B = 200;      // b: hi = min(8192,200) = 200 -> scan IN[0..199]
    constexpr uint32_t TILE = 8192;

    const uint32_t inBase  = 2 * B + 2;            // 402
    const uint32_t outBase = C * TILE + 2 * B + 2; // 8594

    const uint32_t BUFN = outBase + B + 64;        // covers OUT[0..B]
    static uint32_t buf[8594 + 200 + 64];
    static uint32_t ref[8594 + 200 + 64];
    std::memset(buf, 0, sizeof(buf));

    // IN[0..B-1]: irregular per-block local counts.
    for (uint32_t k = 0; k < B; ++k) {
        uint32_t v = 1 + ((k * 1103515245u + 12345u) >> 24) % 7;  // 1..7, irregular
        buf[inBase + k] = v;
    }
    // poison OUT so a "no-write" bug is caught
    for (uint32_t j = 0; j <= B; ++j) buf[outBase + j] = 0xDEADBEEF;
    std::memcpy(ref, buf, sizeof(uint32_t) * BUFN);

    // Independent reference: exclusive prefix sum of IN into OUT.
    uint32_t acc = 0;
    for (uint32_t j = 0; j <= B; ++j) {
        ref[outBase + j] = acc;            // OUT[j] = sum_{k<j} IN[k]
        if (j < B) acc += ref[inBase + j];
    }

    launchKernel(_Z41CompressMarkerLabelsSumLocalBlockCountsUFjjjPh,
                 1, 32,                          // 1 block, 1 lane survives
                 ArgAt{0x210}, (uint32_t)A,      // a
                 ArgAt{0x214}, (uint32_t)B,      // b
                 ArgAt{0x218}, (uint32_t)C,      // c
                 ArgAt{0x220}, (uint32_t*)buf);  // pBuffer

    int bad = 0;
    for (uint32_t i = 0; i < BUFN; ++i)
        if (buf[i] != ref[i]) {
            if (bad < 8) printf("  buf[%u]: got %u want %u\n", i, buf[i], ref[i]);
            ++bad;
        }
    printf("=== Sum: exclusive prefix sum, OUT[B]=%u (total of IN[0..%u]) ===\n",
           buf[outBase + B], B - 1);
    printf("  OUT[0..8] = ");
    for (int j = 0; j <= 8; ++j) printf("%u ", buf[outBase + j]);
    printf("\n");
    printf("mismatches = %d / %u\n", bad, BUFN);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
