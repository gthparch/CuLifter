// host_cpu.cpp — x86 exec-correctness for NPP PBA kernelPropagateInterband.
//
// Kernel: _Z24kernelPropagateInterbandP6short2S0_ii
//   kernelPropagateInterband(short2* input, short2* output, int size, int bandSize)
//   (from libnpp sm_90 npp.cubin, lifted to x86 by SLifter)
//
// Ground-truth (PBA+ pba2DKernel.h, NUS — exact match to the SASS):
//   tx = blockIdx.x*blockDim.x + threadIdx.x;  inc = bandSize*size;
//   ty = blockIdx.y*bandSize;
//   topId    = ty*size + tx;
//   bottomId = (ty+bandSize-1)*size + tx;
//   tid = blockIdx.y*size + tx;
//   bid = tid + size*size/bandSize;
//   // top row, look backward (up) one band at a time
//   pixel = input[topId];  myDist = abs(pixel.y - ty);  output[tid] = pixel;
//   for (nid = bottomId-inc; nid>=0; nid-=inc) {
//     pixel = input[nid];
//     if (pixel.x != MARKER) { if (abs(pixel.y-ty) < myDist) output[tid]=pixel; break; } }
//   // last row, look downward
//   ty = ty+bandSize-1;
//   pixel = input[bottomId];  myDist = abs(pixel.y - ty);  output[bid] = pixel;
//   for (ny=ty+1, nid=topId+inc; ny<size; ny+=bandSize, nid+=inc) {
//     pixel = input[nid];
//     if (pixel.x != MARKER) { if (abs(pixel.y-ty) < myDist) output[bid]=pixel; break; } }
//
// SASS confirms: short2.x = low 16b (PRMT .9910), .y = high 16b (PRMT .bb32);
// MARKER = -32768; abs via IABS; WARPSYNC.ALL is warp-reconvergence only (no
// cross-thread output dependence). Writes hit output[tid]/output[bid], distinct
// per (blockIdx.y, tx) — order-independent. Params input@0x210, output@0x218,
// size@0x220, bandSize@0x224.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "kernel_wrapper.h"

extern "C" void _Z24kernelPropagateInterbandP6short2S0_ii();

struct s2 { int16_t x, y; };
static const int16_t MARKER = -32768;

int main() {
    constexpr int SIZE = 8;        // grid.x*blockDim.x == size
    constexpr int BAND = 2;        // bandSize; nbands = size/bandSize = 4
    constexpr int NBANDS = SIZE / BAND;
    constexpr int N = SIZE * SIZE;
    // output is the band-summary array: NBANDS*size top entries + size*size/bandSize
    // base for the bottom entries. Largest written index is bid_max =
    // (NBANDS-1)*size + (size-1) + size*size/bandSize. Size output generously.
    constexpr int OUTN = 2 * (SIZE * SIZE);   // plenty (covers tid and bid ranges)

    s2 in[N], out[OUTN], ref[OUTN];
    for (int i = 0; i < OUTN; ++i) { out[i] = {0x0AAA, 0x0BBB}; ref[i] = {0x0AAA, 0x0BBB}; }

    auto TOID = [](int x,int y){ return y*SIZE + x; };
    // Discriminating input: distinct site-y values; some MARKER blanks so the
    // backward/forward scan loops actually fire and the abs()-distance compare
    // selects a closer neighbour in some lanes but not others.
    for (int tx = 0; tx < SIZE; ++tx)
        for (int y = 0; y < SIZE; ++y) {
            s2 p;
            // Column tx==0: a single site at row 5, blanks elsewhere -> propagation.
            // Other columns: a checkerboard of sites/blanks with distinct .y.
            bool site = ((tx + y) % 3 != 0);
            if (tx == 0) site = (y == 5);
            if (site) { p.x = (int16_t)(tx*10 + 1); p.y = (int16_t)(y*2 + tx); }   // x!=MARKER
            else      { p.x = MARKER;               p.y = (int16_t)(100 + y*3 + tx); }
            in[TOID(tx,y)] = p;
        }

    // ---- Independent reference ----
    for (int byy = 0; byy < NBANDS; ++byy) {
      for (int tx = 0; tx < SIZE; ++tx) {
        int inc = BAND * SIZE;
        int ty = byy * BAND;
        int topId = ty*SIZE + tx;
        int bottomId = (ty + BAND - 1)*SIZE + tx;
        int tid = byy*SIZE + tx;
        int bid = tid + (SIZE*SIZE)/BAND;
        s2 pixel = in[topId];
        int myDist = abs((int)pixel.y - ty);
        ref[tid] = pixel;
        for (int nid = bottomId - inc; nid >= 0; nid -= inc) {
            pixel = in[nid];
            if (pixel.x != MARKER) { if (abs((int)pixel.y - ty) < myDist) ref[tid] = pixel; break; }
        }
        int ty2 = ty + BAND - 1;
        pixel = in[bottomId];
        myDist = abs((int)pixel.y - ty2);
        ref[bid] = pixel;
        for (int ny = ty2 + 1, nid = topId + inc; ny < SIZE; ny += BAND, nid += inc) {
            pixel = in[nid];
            if (pixel.x != MARKER) { if (abs((int)pixel.y - ty2) < myDist) ref[bid] = pixel; break; }
        }
      }
    }

    launchKernel2D(_Z24kernelPropagateInterbandP6short2S0_ii,
                   Dim3{1, NBANDS}, Dim3{SIZE, 1},
                   ArgAt{0x210}, (void*)in,
                   ArgAt{0x218}, (void*)out,
                   ArgAt{0x220}, (int)SIZE,
                   ArgAt{0x224}, (int)BAND);

    int bad = 0;
    for (int i = 0; i < OUTN; ++i)
        if (out[i].x != ref[i].x || out[i].y != ref[i].y) {
            if (bad < 8) printf("  mismatch[%d]: got(%d,%d) want(%d,%d)\n",
                                i, out[i].x, out[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelPropagateInterband ===\n");
    printf("mismatches = %d / %d\n", bad, OUTN);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
