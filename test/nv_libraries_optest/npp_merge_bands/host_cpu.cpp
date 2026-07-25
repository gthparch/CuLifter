// host_cpu.cpp — x86 exec-correctness for NPP PBA kernelMergeBands.
//
// Kernel: _Z16kernelMergeBandsP6short2S0_S0_ii
//   kernelMergeBands(short2* color, short2* link, short2* output, int size, int bandSize)
//   (from libnpp sm_90 npp.cubin, lifted to x86 by SLifter)
//
// Ground-truth: PBA+ pba2DKernel.h (NUS), transcribed VERBATIM below into the host
// reference (the SASS is a faithful compile of exactly this source — IMAD.WIDE.U32
// + ISETP.GT.AND.EX confirm the 64-bit `dominate` predicate). The reference is
// computed from the ORIGINAL input arrays, independent of the lifted IR.
//
// Real PBA usage (pba2DHost.cu line 123) aliases link == output == pbaTextures[0];
// color == pbaTextures[1] is separate. We mirror that aliasing so within-thread
// read-after-write on `link`/`output` matches. Each thread owns column tx and only
// touches rows in its own column, so there is NO cross-thread conflict — the result
// is order-independent and valid under the wrapper's sequential-CTA model.
//
// short2.x = low 16b, .y = high 16b; MARKER = -32768. Params: color@0x210,
// link@0x218, output@0x220, size@0x228, bandSize@0x22c. grid=(size/bx, noBand/2),
// block=(size). blockIdx.y selects the band pair (band1=2*by, band2=band1+1).
#include <cstdint>
#include <cstdio>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void _Z16kernelMergeBandsP6short2S0_S0_ii();

struct s2 { int16_t x, y; };
static const int16_t MARKER = -32768;
static inline s2 mk(int x, int y){ return s2{ (int16_t)x, (int16_t)y }; }

// ---- verbatim dominate() (64-bit) ----
#define LL long long
static bool dominate(LL x1, LL y1, LL x2, LL y2, LL x3, LL y3, LL x0){
    LL k1 = y2 - y1, k2 = y3 - y2;
    return (k1*(y1+y2) + (x2-x1)*((x1+x2)-(x0<<1)))*k2 >
           (k2*(y2+y3) + (x3-x2)*((x2+x3)-(x0<<1)))*k1;
}
#undef LL

static int SIZE, BAND;
static inline int TOID(int x,int y){ return y*SIZE + x; }

// ---- verbatim kernelMergeBands replayed on the host for one (tx, by) ----
static void ref_merge(int tx, int by, const s2* color, s2* link_out){
    // link and output alias the same buffer (link_out), matching real usage.
    auto LINK = [&](int id)->s2{ return link_out[id]; };
    auto COLOR = [&](int id)->s2{ return color[id]; };

    int band1 = by * 2;
    int band2 = band1 + 1;
    int firsty, lasty;
    s2 last1{0,0}, last2, current;

    lasty = band2*BAND - 1;
    last2 = mk(COLOR(TOID(tx,lasty)).x, LINK(TOID(tx,lasty)).y);
    if (last2.x == MARKER) {
        lasty = last2.y;
        if (lasty >= 0) last2 = mk(COLOR(TOID(tx,lasty)).x, LINK(TOID(tx,lasty)).y);
        else            last2 = mk(MARKER, MARKER);
    }
    if (last2.y >= 0)
        last1 = mk(COLOR(TOID(tx,last2.y)).x, LINK(TOID(tx,last2.y)).y);

    firsty = band2*BAND;
    current = mk(LINK(TOID(tx,firsty)).x, COLOR(TOID(tx,firsty)).x);
    if (current.y == MARKER) {
        firsty = current.x;
        if (firsty >= 0) current = mk(LINK(TOID(tx,firsty)).x, COLOR(TOID(tx,firsty)).x);
        else             current = mk(MARKER, MARKER);
    }

    int top = 0;
    while (top < 2 && current.y >= 0) {
        while (last2.y >= 0) {
            if (!dominate(last1.x, last2.y, last2.x, lasty, current.y, firsty, tx)) break;
            lasty = last2.y; last2 = last1; top--;
            if (last1.y >= 0) last1 = mk(COLOR(TOID(tx,last1.y)).x, LINK(TOID(tx,last1.y)).y);
        }
        link_out[TOID(tx,firsty)] = mk(current.x, lasty);               // output write
        if (lasty >= 0) link_out[TOID(tx,lasty)] = mk(firsty, last2.y); // output write
        last1 = last2; last2 = mk(current.y, lasty); lasty = firsty;
        firsty = current.x;
        top = std::max(1, top + 1);
        if (firsty >= 0) current = mk(LINK(TOID(tx,firsty)).x, COLOR(TOID(tx,firsty)).x);
        else             current = mk(MARKER, MARKER);
    }

    firsty = band1*BAND;
    lasty  = band2*BAND;
    current = LINK(TOID(tx,firsty));
    if (current.y == MARKER && current.x < 0) {
        last1 = LINK(TOID(tx,lasty));
        if (last1.y == MARKER) current.x = last1.x; else current.x = lasty;
        link_out[TOID(tx,firsty)] = current;
    }

    firsty = band1*BAND + BAND - 1;
    lasty  = band2*BAND + BAND - 1;
    current = LINK(TOID(tx,lasty));
    if (current.x == MARKER && current.y < 0) {
        last1 = LINK(TOID(tx,firsty));
        if (last1.x == MARKER) current.y = last1.y; else current.y = firsty;
        link_out[TOID(tx,lasty)] = current;
    }
}

int main(){
    SIZE = 8; BAND = 2;                  // noBand = size/BAND = 4 -> grid.y = noBand/2 = 2
    const int NPAIR = (SIZE / BAND) / 2;
    const int N = SIZE * SIZE;

    // color (separate buffer) and link==output (aliased buffer, two copies: dev & ref)
    s2 color[N], link_dev[N], link_ref[N];

    auto TID = [&](int x,int y){ return y*SIZE + x; };
    // Build a valid two-band doubly-linked proximate-point structure per column.
    // color[id].x = site x-coord (or MARKER for blank rows).
    // link[id].x = forward ptr (next y), link[id].y = backward ptr (prev y); MARKER ends.
    // Lay 2 sites per band so the merge's dominance test runs across the band boundary.
    for (int tx = 0; tx < SIZE; ++tx) {
        for (int y = 0; y < SIZE; ++y) { color[TID(tx,y)] = mk(MARKER, MARKER);
                                         link_dev[TID(tx,y)] = mk(MARKER, MARKER); }
        // sites at rows {0,1} in band0, {2,3} in band1 for the first pair (by=0),
        // and {4,5},{6,7} for the second pair (by=1). Distinct x-coords per row/col.
        for (int pair = 0; pair < (SIZE/BAND)/1; ++pair) {}
        // Place a chained list: rows r0<r1<r2<r3 are sites; link forward/backward.
        int rows[8]; int nr = 0;
        for (int y = 0; y < SIZE; ++y) rows[nr++] = y;   // every row a site -> rich merge
        for (int k = 0; k < nr; ++k) {
            int y = rows[k];
            // distinct, non-monotone site x so dominate() actually discriminates
            int xc = ((tx*3 + y*5 + 7) % 17) - 8;        // in [-8,8], !=MARKER
            color[TID(tx,y)] = mk(xc, y);
            int fwd = (k+1 < nr) ? rows[k+1] : MARKER;   // forward ptr
            int bwd = (k-1 >= 0) ? rows[k-1] : MARKER;   // backward ptr
            link_dev[TID(tx,y)] = mk(fwd, bwd);
        }
    }
    for (int i = 0; i < N; ++i) link_ref[i] = link_dev[i];

    // ---- reference: replay per (tx, by) on link_ref ----
    for (int by = 0; by < NPAIR; ++by)
        for (int tx = 0; tx < SIZE; ++tx)
            ref_merge(tx, by, color, link_ref);

    // ---- device: link_dev is both `link` and `output` (aliased) ----
    launchKernel2D(_Z16kernelMergeBandsP6short2S0_S0_ii,
                   Dim3{1, NPAIR}, Dim3{SIZE, 1},
                   ArgAt{0x210}, (void*)color,     // color
                   ArgAt{0x218}, (void*)link_dev,  // link
                   ArgAt{0x220}, (void*)link_dev,  // output (== link, aliased)
                   ArgAt{0x228}, (int)SIZE,
                   ArgAt{0x22c}, (int)BAND);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (link_dev[i].x != link_ref[i].x || link_dev[i].y != link_ref[i].y) {
            if (bad < 8) printf("  mismatch[%d] (x=%d,y=%d): got(%d,%d) want(%d,%d)\n",
                                i, i%SIZE, i/SIZE, link_dev[i].x, link_dev[i].y,
                                link_ref[i].x, link_ref[i].y);
            ++bad;
        }
    printf("=== kernelMergeBands ===\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
