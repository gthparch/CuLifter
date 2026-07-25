// host_cpu.cpp — x86 exec-correctness for a REAL NPP connected-components kernel.
//
// Kernel: CompressMarkerLabelsPass3<uint,uint>(uint* pSrc, size_t nSrcStep,
//             uint* pDst, size_t nDstStep, NppiSize oSize, int nMaxLabel,
//             uint8_t* pBuffer)
//   _Z25CompressMarkerLabelsPass3IjjEvPT_mPT0_m8NppiSizeiPh  (libnpp sm_90)
//
// Decoded from the SASS (full dump, @P0-predicated instrs included):
//   col = blockIdx.x*blockDim.x + threadIdx.x      (2D launch)
//   row = blockIdx.y*blockDim.y + threadIdx.y
//   if (col >= oSize.width || row >= oSize.height) return;    // oSize @0x230/0x234
//   srcPitch  = nSrcStep / 4                                  // nSrcStep @0x218
//   label = pSrc[row*srcPitch + col]                          // pSrc @0x210
//   if (label >= nMaxLabel) return;                           // nMaxLabel @0x238 (NO write)
//   dstPitch = nDstStep / 4                                   // nDstStep @0x228
//   destIdx  = row*dstPitch + col                             // pDst @0x220
//   if (label == 0)  pDst[destIdx] = 0;
//   else             pDst[destIdx] = ((uint*)pBuffer)[label + nMaxLabel + 2];
//
// The gather offset: SASS forms R4 = label + (nMaxLabel+1), LEA *4 + pBuffer,
// then LDG [.. + 0x4] -> byte (label+nMaxLabel+2)*4 = pBuffer[label+nMaxLabel+2].
// This is the label-remap/gather pass: each pixel's old label is replaced by the
// compacted new label looked up in the prefix-sum region of pBuffer. Output is a
// pure function of (pSrc, pBuffer) — order-independent (each thread writes its own
// disjoint destIdx).
//
// Discriminating input: distinct labels, some 0 (-> 0), some >= nMaxLabel
// (-> left untouched / poison preserved), valid labels gathered to distinct values.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void _Z25CompressMarkerLabelsPass3IjjEvPT_mPT0_m8NppiSizeiPh();

int main() {
    constexpr int W = 20, H = 12;
    constexpr int BX = 8, BY = 4;
    const int GX = (W + BX - 1) / BX;      // 3
    const int GY = (H + BY - 1) / BY;      // 3 (over-covers; bounds guard)
    const int N = W * H;                   // 240 pixels
    const uint32_t PITCH = W;
    const int MAXLABEL = 64;               // nMaxLabel

    static uint32_t src[N], dst[N], ref[N];
    // pBuffer: remap table region is read at index [label + MAXLABEL + 2].
    // Make it big enough for label up to MAXLABEL-1 -> index up to 2*MAXLABEL+1.
    constexpr int BUFN = 4 * 64 + 16;
    static uint32_t buf[BUFN];

    // Remap table: distinct, recognizable new labels for each old label.
    std::memset(buf, 0, sizeof(buf));
    for (int L = 1; L < MAXLABEL; ++L)
        buf[L + MAXLABEL + 2] = (uint32_t)(1000 + L * 3);   // new compacted label

    // Source labels: mix of 0, valid (<MAXLABEL), and out-of-range (>=MAXLABEL).
    constexpr uint32_t POISON = 0xCAFEBABE;
    for (int i = 0; i < N; ++i) {
        if (i % 9 == 0)        src[i] = 0;                       // -> writes 0
        else if (i % 13 == 0)  src[i] = MAXLABEL + (i % 4);     // >= MAXLABEL -> no write
        else                   src[i] = (uint32_t)(i % (MAXLABEL - 1)) + 1; // 1..MAXLABEL-1
        dst[i] = POISON;
    }

    // Independent reference.
    for (int row = 0; row < H; ++row)
      for (int col = 0; col < W; ++col) {
        int idx = row * PITCH + col;
        uint32_t label = src[idx];
        if ((int)label >= MAXLABEL) { ref[idx] = POISON; continue; } // untouched
        if (label == 0)             { ref[idx] = 0; continue; }
        ref[idx] = buf[label + MAXLABEL + 2];
      }

    launchKernel2D(_Z25CompressMarkerLabelsPass3IjjEvPT_mPT0_m8NppiSizeiPh,
                   Dim3{GX, GY}, Dim3{BX, BY},
                   ArgAt{0x210}, (uint32_t*)src,           // pSrc
                   ArgAt{0x218}, (uint64_t)(PITCH * 4),    // nSrcStep
                   ArgAt{0x220}, (uint32_t*)dst,           // pDst
                   ArgAt{0x228}, (uint64_t)(PITCH * 4),    // nDstStep
                   ArgAt{0x230}, (int)W,                   // oSize.width
                   ArgAt{0x234}, (int)H,                   // oSize.height
                   ArgAt{0x238}, (int)MAXLABEL,            // nMaxLabel
                   ArgAt{0x240}, (uint32_t*)buf);          // pBuffer

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (dst[i] != ref[i]) {
            if (bad < 8) printf("  dst[%d]: got %u want %u (src=%u)\n",
                                i, dst[i], ref[i], src[i]);
            ++bad;
        }
    printf("=== Pass3: remap/gather over %d pixels ===\n", N);
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
