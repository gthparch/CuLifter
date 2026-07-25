// host_cpu.cpp — x86 exec-correctness for a REAL npp PBA distance-transform kernel.
//
// Kernel: npp::kernelProximatePoints(short2* input, short2* stack, int size, int bandSize)
//   _Z21kernelProximatePointsP6short2S0_ii  (from libnpp sm_90 npp.cubin)
//
// PBA phase-2 "proximate points": for each column tx, walk the band's rows and
// maintain a stack of the sites that can be the nearest along that column,
// discarding (popping) sites dominated by their neighbours via the integer
// cross-product test dominate(). The stack is stored back into `stack` with each
// entry holding (.x = site x-coord, .y = backward pointer = previous site's row).
// A tail sentinel (.x = MARKER, .y = last site's row) is written at band bottom.
//
// Decoded from full SASS:
//   tx = blockIdx.x*blockDim.x + threadIdx.x       (column)
//   ty = blockIdx.y * bandSize  (mul24-trunc),  id = ty*size + tx
//   lasty=-1; last1.y=-1; last2.y=-1;
//   for (i=0; i<bandSize; ++i, id+=size):
//     current = input[id]                          // .x via PRMT 0x9910, .y via 0xbb32
//     if (current.x != MARKER):
//        while (last2.y >= 0):
//           if (!dominate(last1.x,last2.y, last2.x,lasty, current.x,current.y, tx)) break;
//           lasty = last2.y; last2 = last1;
//           if (last1.y >= 0) last1 = stack[tx + last1.y*size];     // @P0 LDC c[0x218]
//        last1 = last2; last2 = (current.x, lasty); lasty = current.y;
//        stack[id] = last2;                          // LEA into c[0x218]
//   if (lasty != ty+bandSize-1)
//        stack[tx + (ty+bandSize-1)*size] = (MARKER, lasty);   // tail sentinel
//
//   dominate (64-bit, matches the IMAD.WIDE.U32 / IADD3.X chains in SASS):
//     LL k1=y2-y1, k2=y3-y2;
//     return (k1*(y1+y2) + (x2-x1)*((x1+x2)-2*x0))*k2
//          > (k2*(y2+y3) + (x3-x2)*((x2+x3)-2*x0))*k1;
//
// Each column is independent; a thread only re-reads stack rows it wrote earlier in
// the same loop -> pure function of `input` per column, ORDER-INDEPENDENT. Verifiable.
// Params: input@0x210, stack@0x218, size@0x220, bandSize@0x224.
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z21kernelProximatePointsP6short2S0_ii();

struct short2 { int16_t x, y; };
static const int16_t MARKER = (int16_t)0x8000;   // -32768

typedef long long LL;
static bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0) {
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1 * (y1 + y2) + (x2 - x1) * ((x1 + x2) - (x0 << 1))) * k2
         > (k2 * (y2 + y3) + (x3 - x2) * ((x2 + x3) - (x0 << 1))) * k1;
}

int main() {
    constexpr int SIZE = 32;
    constexpr int BAND = 16;             // one band per column makes the stack deep
    constexpr int NBAND = SIZE / BAND;   // 2 bands
    constexpr int N = SIZE * SIZE;
    constexpr int BLOCK = SIZE;

    short2 in[N], st[N], ref[N];

    // Discriminating input: several columns with multiple sites at varying x so
    // dominate() actually pops mid-sites. Sites placed so some are dominated.
    for (int i = 0; i < N; ++i) { in[i].x = MARKER; in[i].y = MARKER; }
    auto put = [&](int x, int y, int16_t sx, int16_t sy){ in[y*SIZE + x].x = sx; in[y*SIZE + x].y = sy; };
    // column 5 (band0 rows0..15): sites whose middle one is dominated by neighbours
    put(5, 1,  5, 1);     // site x=5
    put(5, 4,  9, 4);     // site x=9  (farther in column-x -> may be dominated)
    put(5, 7,  5, 7);     // site x=5
    put(5, 12, 6, 12);    // site x=6
    // column 12: monotone sites (none dominated)
    put(12, 2,  10, 2);
    put(12, 9,  11, 9);
    put(12, 14, 12, 14);
    // column 20: two sites, then band2
    put(20, 3,  20, 3);
    put(20, 10, 25, 10);
    put(20, 18, 22, 18);   // band1 (rows16..31)
    put(20, 28, 28, 28);
    // column 31 last
    put(31, 0,  31, 0);
    put(31, 15, 30, 15);

    for (int i = 0; i < N; ++i) { st[i].x = 0x1234; st[i].y = 0x5678; }   // poison

    // Independent reference: replicate the exact stack algorithm into ref[].
    for (int i = 0; i < N; ++i) ref[i] = st[i];
    for (int b = 0; b < NBAND; ++b) {
        int ty = b * BAND;
        for (int tx = 0; tx < SIZE; ++tx) {
            int lasty = -1;
            short2 last1{0,-1}, last2{0,-1};
            int id = ty * SIZE + tx;
            for (int i = 0; i < BAND; ++i, id += SIZE) {
                short2 cur = in[id];
                if (cur.x != MARKER) {
                    while (last2.y >= 0) {
                        if (!dominate(last1.x, last2.y, last2.x, lasty, cur.x, cur.y, tx)) break;
                        lasty = last2.y; last2 = last1;
                        if (last1.y >= 0) last1 = ref[tx + last1.y * SIZE];
                    }
                    last1 = last2;
                    last2 = short2{cur.x, (int16_t)lasty};
                    lasty = cur.y;
                    ref[id] = last2;
                }
            }
            int tailRow = ty + BAND - 1;
            if (lasty != tailRow)
                ref[tx + tailRow * SIZE] = short2{MARKER, (int16_t)lasty};
        }
    }

    launchKernel2D(_Z21kernelProximatePointsP6short2S0_ii,
                   Dim3{SIZE / BLOCK, NBAND}, Dim3{BLOCK, 1},
                   ArgAt{0x210}, (short2*)in,
                   ArgAt{0x218}, (short2*)st,
                   ArgAt{0x220}, (int)SIZE,
                   ArgAt{0x224}, (int)BAND);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (st[i].x != ref[i].x || st[i].y != ref[i].y) {
            if (bad < 12) printf("  mismatch[%d] (col %d,row %d): got (%d,%d) want (%d,%d)\n",
                                 i, i % SIZE, i / SIZE, st[i].x, st[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelProximatePoints: col5 stack entries (got) ===\n  ");
    for (int r = 0; r < BAND; ++r) printf("(%d,%d) ", st[r*SIZE+5].x, st[r*SIZE+5].y);
    printf("\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
