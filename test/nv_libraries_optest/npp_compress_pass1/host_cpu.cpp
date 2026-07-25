// host_cpu.cpp — x86 exec-correctness for a REAL NPP connected-components kernel.
//
// Kernel: CompressMarkerLabelsPass1<uint,uint>(uint* pSrc, size_t nSrcStep,
//             uint* pDst, size_t nDstStep, NppiSize oSize, int nMaxLabel,
//             uint8_t* pBuffer)
//   _Z25CompressMarkerLabelsPass1IjjEvPT_mPT0_m8NppiSizeiPh  (libnpp sm_90)
//
// Decoded from the SASS (full dump, predicated instrs included):
//   col = blockIdx.x*blockDim.x + threadIdx.x      (2D launch)
//   row = blockIdx.y*blockDim.y + threadIdx.y
//   if (col >= oSize.width || row >= oSize.height) return;   // oSize @0x230/0x234
//   pitch = nSrcStep / 4                                     // nSrcStep @0x218 (bytes)
//   label = pSrc[row*pitch + col]                            // pSrc @0x210
//   if (label == 0 || label >= nMaxLabel) return;            // nMaxLabel @0x238
//   ((uint*)pBuffer)[label + 1] = 1;                         // pBuffer @0x240
//
// The store is `STG [base + value*4 + 0x4]` with base = c[0x240]: byte off
// (label+1)*4 → pBuffer[label+1] = 1. This is the "mark which labels are used"
// scatter. Every active thread writes the SAME value 1, so the result is a pure,
// order-independent function of the inputs (idempotent scatter) — matches the
// wrapper's sequential CTAs.
//
// Discriminating input: distinct label per pixel, with some zeros (skipped),
// some labels >= nMaxLabel (skipped), and valid labels in [1,nMaxLabel) marked.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void _Z25CompressMarkerLabelsPass1IjjEvPT_mPT0_m8NppiSizeiPh();

int main() {
    constexpr int W = 24, H = 10;          // image 24x10
    constexpr int BX = 8, BY = 4;          // block 8x4
    const int GX = (W + BX - 1) / BX;      // 3
    const int GY = (H + BY - 1) / BY;      // 3 (over-covers; @P0 EXIT bounds guard)
    const int N = W * H;                   // 240 pixels
    const uint32_t PITCH = W;              // elements per row (nSrcStep = W*4 bytes)
    const int MAXLABEL = 100;              // nMaxLabel

    // pBuffer holds the mark region [1..MAXLABEL]; size generously, zero-init.
    constexpr int BUFN = 512;
    static uint32_t buf[BUFN], ref[BUFN];
    static uint32_t src[N];

    std::memset(buf, 0, sizeof(buf));
    std::memset(ref, 0, sizeof(ref));

    // Discriminating labels: mix of 0, valid (<MAXLABEL), and out-of-range.
    for (int i = 0; i < N; ++i) {
        if (i % 7 == 0)        src[i] = 0;                  // background -> skipped
        else if (i % 11 == 0)  src[i] = MAXLABEL + (i % 5); // >= nMaxLabel -> skipped
        else                   src[i] = (uint32_t)(i % (MAXLABEL - 1)) + 1; // 1..MAXLABEL-1
    }

    // Independent reference: replicate the scatter.
    for (int row = 0; row < H; ++row)
      for (int col = 0; col < W; ++col) {
        uint32_t label = src[row * PITCH + col];
        if (label == 0 || (int)label >= MAXLABEL) continue;
        ref[label + 1] = 1;
      }

    launchKernel2D(_Z25CompressMarkerLabelsPass1IjjEvPT_mPT0_m8NppiSizeiPh,
                   Dim3{GX, GY}, Dim3{BX, BY},
                   ArgAt{0x210}, (uint32_t*)src,           // pSrc
                   ArgAt{0x218}, (uint64_t)(PITCH * 4),    // nSrcStep (bytes)
                   ArgAt{0x230}, (int)W,                   // oSize.width
                   ArgAt{0x234}, (int)H,                   // oSize.height
                   ArgAt{0x238}, (int)MAXLABEL,            // nMaxLabel
                   ArgAt{0x240}, (uint32_t*)buf);          // pBuffer

    int bad = 0;
    for (int i = 0; i < BUFN; ++i)
        if (buf[i] != ref[i]) {
            if (bad < 8) printf("  buf[%d]: got %u want %u\n", i, buf[i], ref[i]);
            ++bad;
        }
    int marked = 0; for (int i = 0; i < BUFN; ++i) marked += (ref[i] != 0);
    printf("=== Pass1: %d marked labels out of buffer size %d ===\n", marked, BUFN);
    printf("mismatches = %d / %d\n", bad, BUFN);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
