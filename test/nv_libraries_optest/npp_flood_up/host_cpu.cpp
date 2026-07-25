// host_cpu.cpp — x86 exec-correctness for a REAL npp PBA distance-transform kernel.
//
// Kernel: npp::kernelFloodUp(short2* input, short2* output, int size, int bandSize)
//   _Z13kernelFloodUpP6short2S0_ii  (from libnpp sm_90 npp.cubin)
//
// PBA phase-1 "flood up". Each thread owns one column tx and walks bandSize rows
// UPWARD within its band starting from the band's bottom row. It carries the
// nearest-by-row site seen so far. short2 packs a site coord; .x==MARKER means empty.
//
// Decoded from full SASS (note the carry is kept split: pixel1.x and pixel1.y):
//   tx = blockIdx.x*blockDim.x + threadIdx.x
//   ty = (blockIdx.y+1)*bandSize - 1   (mul24-trunc; fine here)   -- band bottom row
//   id = ty*size + tx
//   pixel1 = (MARKER, MARKER)
//   for (i=0; i<bandSize; ++i, id-=size):
//       dist1 = abs(pixel1.y + i - ty)          // PRMT .y of carry  (R16 low16)
//       pixel2 = input[id]
//       dist2 = abs(pixel2.y + i - ty)          // PRMT 0xbb32 = pixel2.y
//       if (dist2 < dist1) pixel1 = pixel2      // SEL on !(dist2>=dist1)
//       output[id] = pixel1                     // PRMT 0x5410 pack (x,y)
//   abs(pixel.y + i - ty) == abs(pixel.y - currentRow), currentRow = ty - i.
//
// Per-column independent, pure function of input -> ORDER-INDEPENDENT, verifiable.
// Params: input@0x210, output@0x218, size@0x220, bandSize@0x224.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "kernel_wrapper.h"

extern "C" void _Z13kernelFloodUpP6short2S0_ii();

struct short2 { int16_t x, y; };
static const int16_t MARKER = (int16_t)0x8000;   // -32768

int main() {
    constexpr int SIZE = 32;
    constexpr int BAND = 8;
    constexpr int NBAND = SIZE / BAND;
    constexpr int N = SIZE * SIZE;
    constexpr int BLOCK = SIZE;

    short2 in[N], out[N], ref[N];

    for (int i = 0; i < N; ++i) { in[i].x = MARKER; in[i].y = MARKER; }
    auto put = [&](int x, int y, int16_t sx, int16_t sy){ in[y*SIZE + x].x = sx; in[y*SIZE + x].y = sy; };
    // column 3: sites at rows 1, 5 (band0) and 20 (band2) — flood-up carries the
    //   nearest-below site up, so rows above a site in the same band take it only
    //   while it is the closest; the abs() makes it a true nearest-by-row choice.
    put(3, 1,   100, 1);
    put(3, 5,   101, 5);
    put(3, 20,  120, 20);
    put(10, 6,  210, 6);     // column 10: lone site row 6
    put(17, 0,  300, 0);     // column 17: rows 0 and 9
    put(17, 9,  309, 9);
    put(31, 31, 431, 31);    // column 31: row 31
    for (int i = 0; i < N; ++i) { out[i].x = 0x1234; out[i].y = 0x5678; }

    for (int i = 0; i < N; ++i) ref[i] = out[i];
    for (int b = 0; b < NBAND; ++b) {
        int ty = (b + 1) * BAND - 1;        // band bottom row
        for (int tx = 0; tx < SIZE; ++tx) {
            short2 p1{MARKER, MARKER};
            int id = ty * SIZE + tx;
            for (int i = 0; i < BAND; ++i, id -= SIZE) {
                int dist1 = abs((int)p1.y + i - ty);
                short2 p2 = in[id];
                int dist2 = abs((int)p2.y + i - ty);
                if (dist2 < dist1) p1 = p2;
                ref[id] = p1;
            }
        }
    }

    launchKernel2D(_Z13kernelFloodUpP6short2S0_ii,
                   Dim3{SIZE / BLOCK, NBAND}, Dim3{BLOCK, 1},
                   ArgAt{0x210}, (short2*)in,
                   ArgAt{0x218}, (short2*)out,
                   ArgAt{0x220}, (int)SIZE,
                   ArgAt{0x224}, (int)BAND);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (out[i].x != ref[i].x || out[i].y != ref[i].y) {
            if (bad < 8) printf("  mismatch[%d] (col %d,row %d): got (%d,%d) want (%d,%d)\n",
                                i, i % SIZE, i / SIZE, out[i].x, out[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelFloodUp: col3 rows0..7 (got) ===\n  ");
    for (int r = 0; r < 8; ++r) printf("(%d,%d) ", out[r*SIZE+3].x, out[r*SIZE+3].y);
    printf("\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
