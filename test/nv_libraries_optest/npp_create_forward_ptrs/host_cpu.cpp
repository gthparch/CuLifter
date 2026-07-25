// host_cpu.cpp — x86 exec-correctness for NPP PBA kernelCreateForwardPointers.
//
// Kernel: _Z27kernelCreateForwardPointersP6short2S0_ii
//   kernelCreateForwardPointers(short2* input, short2* output, int size, int bandSize)
//   (from libnpp sm_90 npp.cubin, lifted to x86 by SLifter)
//
// Ground-truth source (PBA+ pba2DKernel.h, NUS — exact match to the SASS):
//   tx   = blockIdx.x*blockDim.x + threadIdx.x;
//   ty   = (blockIdx.y+1)*bandSize - 1;           // last row of this band
//   id   = ty*size + tx;
//   lasty = -1;
//   current = input[id];
//   nexty = (current.x == MARKER) ? current.y : ty;
//   for (i=0; i<bandSize; i++, id-=size)
//     if (ty - i == nexty) {
//       current = make_short2(lasty, input[id].y);   // .x=back-ptr, .y=this row's site-y
//       output[id] = current;
//       lasty = nexty;  nexty = current.y;           // = input[id].y
//     }
//   if (lasty != ty-bandSize+1)
//     output[id+size] = make_short2(lasty, MARKER);  // id now = TOID(tx, ty-bandSize, size)
//
// SASS confirms: short2.x = low 16b (PRMT .9910 sign-extend), .y = high 16b
// (PRMT .bb32/.7632); MARKER = -32768 = 0x8000; input.y read as LDG.E.U16 +0x2.
// Params: input@0x210, output@0x218, size@0x220, bandSize@0x224.
//
// Output writes are at distinct id per (tx,band) — no cross-thread conflict; the
// kernel only reads `input` and writes `output`, so it is order-independent.
// Reference computed independently from the ORIGINAL input array below.
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z27kernelCreateForwardPointersP6short2S0_ii();

struct s2 { int16_t x, y; };
static const int16_t MARKER = -32768;

int main() {
    constexpr int SIZE = 8;        // grid.x*blockDim.x == size (one thread per column)
    constexpr int BAND = 4;        // bandSize; nbands = size/bandSize = 2
    constexpr int NBANDS = SIZE / BAND;
    constexpr int N = SIZE * SIZE;

    s2 in[N], out[N], ref[N];
    // Poison output so any un-written cell is detectable, and seed ref identically.
    for (int i = 0; i < N; ++i) { out[i] = {0x1234, 0x5678}; ref[i] = {0x1234, 0x5678}; }

    // Discriminating input: a doubly-linked-list embedding inside each band column.
    // For each column tx, lay sites at a few rows and MARKER (blank, with a backward
    // pointer in .y) elsewhere — so the "if (ty-i == nexty)" walk visits a nontrivial
    // subset and chains lasty/nexty. Distinct .y values per row catch byte/order bugs.
    auto TOID = [](int x, int y){ return y*SIZE + x; };
    for (int tx = 0; tx < SIZE; ++tx) {
        for (int y = 0; y < SIZE; ++y) {
            s2 p;
            // sites at rows {band*BAND, band*BAND+2}; MARKER chains the rest down.
            int band = y / BAND, r = y % BAND;
            if (r == 0 || r == 2) { p.x = (int16_t)(tx + 1);              // site: x!=MARKER
                                    p.y = (int16_t)(y); }                 // site row coord
            else {
                // blank: .x = MARKER, .y = nexty backward pointer to a real site row.
                // point to the nearest site row at or below within this band's tail logic:
                int tgt = band*BAND + (r > 2 ? 2 : 0);
                p.x = MARKER; p.y = (int16_t)(band*BAND + 0 == y ? y : tgt);
                // make the tail row (r==3) point to row band*BAND+2 so the head walk runs
                if (r == 3) p.y = (int16_t)(band*BAND + 2);
                if (r == 1) p.y = (int16_t)(band*BAND + 0);
            }
            in[TOID(tx,y)] = p;
        }
    }

    // ---- Independent reference: replay the kernel loop per (tx, band) ----
    for (int by = 0; by < NBANDS; ++by) {
      for (int tx = 0; tx < SIZE; ++tx) {
        int ty = (by + 1) * BAND - 1;
        int id = ty * SIZE + tx;
        int lasty = -1, nexty;
        s2 cur = in[id];
        nexty = (cur.x == MARKER) ? cur.y : ty;
        for (int i = 0; i < BAND; ++i, id -= SIZE) {
            if (ty - i == nexty) {
                cur.x = (int16_t)lasty;
                cur.y = in[id].y;
                ref[id] = cur;
                lasty = nexty;
                nexty = cur.y;
            }
        }
        if (lasty != ty - BAND + 1)
            ref[id + SIZE] = s2{ (int16_t)lasty, MARKER };
      }
    }

    // 2D launch: grid {1, NBANDS}, block {SIZE, 1}; reads CTAID.Y, no bounds check.
    launchKernel2D(_Z27kernelCreateForwardPointersP6short2S0_ii,
                   Dim3{1, NBANDS}, Dim3{SIZE, 1},
                   ArgAt{0x210}, (void*)in,
                   ArgAt{0x218}, (void*)out,
                   ArgAt{0x220}, (int)SIZE,
                   ArgAt{0x224}, (int)BAND);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (out[i].x != ref[i].x || out[i].y != ref[i].y) {
            if (bad < 8) printf("  mismatch[%d] (x=%d,y=%d): got(%d,%d) want(%d,%d)\n",
                                i, i%SIZE, i/SIZE, out[i].x, out[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelCreateForwardPointers ===\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
