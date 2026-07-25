// host_cpu.cpp — x86 exec-correctness for a REAL npp PBA distance-transform kernel.
//
// Kernel: npp::kernelFloodDown(short2* input, short2* output, int size, int bandSize)
//   _Z15kernelFloodDownP6short2S0_ii  (from libnpp sm_90 npp.cubin)
//
// PBA (Parallel Banding Algorithm) phase-1 "flood down". Each thread owns one
// column tx and walks `bandSize` rows downward within its band, carrying the
// last seen non-empty site downward. short2 packs a 2D site coord; .x == MARKER
// (-32768 = 0x8000) means "empty pixel".
//
// Decoded from full SASS:
//   tx = blockIdx.x*blockDim.x + threadIdx.x           (column)
//   ty = blockIdx.y * bandSize  (mul24-truncated; fine for our small sizes)
//   id = ty*size + tx
//   pixel1 = (MARKER, MARKER)
//   for (i=0; i<bandSize; ++i, id+=size):
//       pixel2 = input[id]
//       if (pixel2.x != MARKER) pixel1 = pixel2     // .x via PRMT 0x9910 (low16, sign-ext)
//       output[id] = pixel1                          // repack via PRMT 0x7610/0x5410
//   bandSize<1 -> EXIT immediately (we always use bandSize>=1).
//
// Pure function of `input`, each column independent and writes only its own
// column rows -> ORDER-INDEPENDENT, verifiable under the sequential-CTA wrapper.
//
// Params (SM90 const bank): input@0x210, output@0x218, size@0x220, bandSize@0x224.
// desc[UR][addr] uses base 0 -> raw device pointers go straight in.
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z15kernelFloodDownP6short2S0_ii();

struct short2 { int16_t x, y; };
static const int16_t MARKER = (int16_t)0x8000;   // -32768

int main() {
    // Square texture: size x size. One thread per column. Bands of bandSize rows.
    constexpr int SIZE = 32;        // columns AND rows
    constexpr int BAND = 8;         // rows per band -> 4 bands (blockIdx.y in 0..3)
    constexpr int NBAND = SIZE / BAND;
    constexpr int N = SIZE * SIZE;
    constexpr int BLOCK = SIZE;     // 32 threads = 32 columns (one CTA-x covers all columns)

    short2 in[N], out[N], ref[N];

    // Discriminating input: scatter a handful of real sites, rest are MARKER.
    // Distinct (x,y) so a wrong byte-pack or wrong carry direction is caught.
    for (int i = 0; i < N; ++i) { in[i].x = MARKER; in[i].y = MARKER; }
    auto put = [&](int x, int y, int16_t sx, int16_t sy){ in[y*SIZE + x].x = sx; in[y*SIZE + x].y = sy; };
    // column 3: sites at rows 1 and 5 (within band0 rows0..7) and row 20 (band2)
    put(3, 1,   100, 1);
    put(3, 5,   101, 5);
    put(3, 20,  120, 20);
    // column 10: site only at row 6 (band0)
    put(10, 6,  210, 6);
    // column 17: site at row 0 (band0 top) and row 9 (band1)
    put(17, 0,  300, 0);
    put(17, 9,  309, 9);
    // column 31 (last): site at row 31 (last band bottom)
    put(31, 31, 431, 31);
    for (int i = 0; i < N; ++i) { out[i].x = 0x1234; out[i].y = 0x5678; }   // poison

    // Independent reference: per band, per column, carry last-seen site downward.
    for (int i = 0; i < N; ++i) ref[i] = out[i];
    for (int b = 0; b < NBAND; ++b) {
        int ty = b * BAND;
        for (int tx = 0; tx < SIZE; ++tx) {
            short2 p1{MARKER, MARKER};
            int id = ty * SIZE + tx;
            for (int i = 0; i < BAND; ++i, id += SIZE) {
                short2 p2 = in[id];
                if (p2.x != MARKER) p1 = p2;
                ref[id] = p1;
            }
        }
    }

    // 2D launch: gridDim.x = SIZE/BLOCK = 1, gridDim.y = NBAND; block = BLOCK threads.
    launchKernel2D(_Z15kernelFloodDownP6short2S0_ii,
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
    printf("=== kernelFloodDown: col3 rows0..7 (got) ===\n  ");
    for (int r = 0; r < 8; ++r) printf("(%d,%d) ", out[r*SIZE+3].x, out[r*SIZE+3].y);
    printf("\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
