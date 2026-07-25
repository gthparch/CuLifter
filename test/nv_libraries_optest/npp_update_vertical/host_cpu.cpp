// host_cpu.cpp — x86 exec-correctness for a REAL npp PBA distance-transform kernel.
//
// Kernel: npp::kernelUpdateVertical(short2* color, short2* margin, short2* output,
//                                   int size, int bandSize)
//   _Z20kernelUpdateVerticalP6short2S0_S0_ii  (from libnpp sm_90 npp.cubin)
//
// PBA phase-1 "update vertical". Within each band, every column picks for each of
// its rows the nearest-by-row site among (its own flooded pixel, the band's `top`
// margin, the band's `bottom` margin), then the whole band tile is TRANSPOSED
// through __shared__ short2 block[64][64] and written to `output` with x/y swapped.
// This kernel uses shared memory + __syncthreads, so its correctness exercises the
// wrapper's intra-CTA concurrent-thread + barrier model (NOT order-independent in
// the trivial sense: thread i's stores are read back by thread tx after the barrier).
//
// Decoded from full SASS:
//   tx = blockIdx.x*blockDim.x + threadIdx.x  (= threadIdx.x here, gridDim.x=1)
//   ty = blockIdx.y * bandSize
//   top    = margin[blockIdx.y*size + tx]
//   bottom = margin[(blockIdx.y + size/bandSize)*size + tx]
//   id = ty*size + tx ; n_step = bandSize / blockDim.x   (= 1 here)
//   for step in [0,n_step):
//     y_start = blockIdx.y*bandSize + step*blockDim.x ;  y_end = y_start + blockDim.x
//     for ty in [y_start, y_end), id += size:
//        pixel = color[id]; myDist = abs(pixel.y - ty)            // PRMT 0xbb32 = .y
//        d = abs(top.y - ty);    if (d<myDist){myDist=d; pixel=top;}
//        d = abs(bottom.y - ty); if (d<myDist) pixel=bottom;
//        block[threadIdx.x][ty - y_start] = make_short2(pixel.y, pixel.x)   // TRANSPOSE
//     __syncthreads()                                            // BAR.SYNC
//     tid = TOID(blockIdx.y*bandSize + step*blockDim.x + threadIdx.x, blockIdx.x*blockDim.x, size)
//     for i in [0,blockDim.x), tid += size:
//        output[tid] = block[i][threadIdx.x]                     // LDS [R27 + i*0x100]
//     __syncthreads()
//
// Params: color@0x210, margin@0x218, output@0x220, size@0x228, bandSize@0x22c.
// We set blockDim.x == bandSize so n_step == 1 (matches the single-step SASS path).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "kernel_wrapper.h"

extern "C" void _Z20kernelUpdateVerticalP6short2S0_S0_ii();

struct short2 { int16_t x, y; };
static const int16_t MARKER = (int16_t)0x8000;   // -32768

int main() {
    // Square texture size x size; band tile B x B (B = blockDim.x = bandSize).
    constexpr int SIZE = 16;
    constexpr int B    = 16;             // blockDim.x == bandSize -> n_step = 1
    constexpr int NBAND = SIZE / B;      // 1 band here (size/bandSize == 1)
    constexpr int N = SIZE * SIZE;

    short2 color[N], output[N], ref[N];
    // margin has size*(size/bandSize + 1) ... kernel indexes margin up to
    // (blockIdx.y + size/bandSize)*size + tx. For NBAND=1, size/bandSize=1, so
    // max index = (0 + 1)*size + tx = size..2*size-1. Allocate 2*size rows.
    constexpr int MARGIN_ROWS = SIZE + (SIZE / B) * SIZE;   // = 2*SIZE safe upper bound
    short2 margin[MARGIN_ROWS * SIZE];

    // ---- discriminating inputs ----
    for (int i = 0; i < N; ++i) { color[i].x = MARKER; color[i].y = MARKER; }
    // give every column a couple of sites at distinct rows so the nearest-by-row
    // pick is non-trivial, and distinct x so the transpose (x<->y) is observable.
    auto putc = [&](int x, int y, int16_t sx, int16_t sy){ color[y*SIZE + x] = short2{sx, sy}; };
    for (int x = 0; x < SIZE; ++x) {
        putc(x, (x % 5),            (int16_t)(100 + x), (int16_t)(x % 5));
        putc(x, (x % 7) + 8,        (int16_t)(200 + x), (int16_t)((x % 7) + 8));
    }
    // margin top/bottom rows: site coords with varied .y so they sometimes win.
    for (int i = 0; i < MARGIN_ROWS * SIZE; ++i) { margin[i].x = MARKER; margin[i].y = MARKER; }
    for (int x = 0; x < SIZE; ++x) {
        margin[0 * SIZE + x] = short2{(int16_t)(300 + x), (int16_t)(x % 3)};        // top (by=0)
        margin[1 * SIZE + x] = short2{(int16_t)(400 + x), (int16_t)(SIZE - 1 - x % 4)}; // bottom
    }
    for (int i = 0; i < N; ++i) output[i] = short2{0x1234, 0x5678};   // poison

    // ---- independent reference (models the transpose explicitly) ----
    // Per-cell nearest-by-row pick P(row, col) over {color[row][col], top[col],
    // bottom[col]}, then write transposed AND x/y-swapped:
    //   output[col][row] = (P.y, P.x)
    // (Derivation: thread t in block (bx,by) handles tx=bx*B+t; tid =
    //  TOID(by*B+t, bx*B, size) = (bx*B)*size + (by*B+t); writes output[tid+i*size]
    //  = block[i][t] = transpose of column (bx*B+i) row (by*B+t). Substituting
    //  output row R=bx*B+i, col C=by*B+t gives source (row C, col R).)
    // Simplest faithful model: directly iterate over thread t and the read-back
    // index i. block[i][t] is thread i's pixel for source (col = bx*B+i, row = by*B+t).
    for (int idx = 0; idx < N; ++idx) ref[idx] = output[idx];
    const int bx = 0;
    for (int by = 0; by < NBAND; ++by) {
        for (int t = 0; t < B; ++t) {                        // thread that writes
            for (int i = 0; i < B; ++i) {                    // read-back index
                int col = bx * B + i;                        // source column = thread i's tx
                int row = by * B + t;                        // source row = local row t
                short2 top    = margin[by * SIZE + col];
                short2 bottom = margin[(by + SIZE / B) * SIZE + col];
                short2 pixel  = color[row * SIZE + col];
                int myDist = abs((int)pixel.y - row);
                int d = abs((int)top.y - row);    if (d < myDist) { myDist = d; pixel = top; }
                d = abs((int)bottom.y - row);     if (d < myDist) pixel = bottom;
                short2 transposed{pixel.y, pixel.x};
                int out_idx = (bx * B + i) * SIZE + (by * B + t);
                ref[out_idx] = transposed;
            }
        }
    }

    // 2D launch: gridDim.x = SIZE/B, gridDim.y = NBAND; block = B threads.
    launchKernel2D(_Z20kernelUpdateVerticalP6short2S0_S0_ii,
                   Dim3{SIZE / B, NBAND}, Dim3{B, 1},
                   ArgAt{0x210}, (short2*)color,
                   ArgAt{0x218}, (short2*)margin,
                   ArgAt{0x220}, (short2*)output,
                   ArgAt{0x228}, (int)SIZE,
                   ArgAt{0x22c}, (int)B);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (output[i].x != ref[i].x || output[i].y != ref[i].y) {
            if (bad < 12) printf("  mismatch[%d] (r%d,c%d): got (%d,%d) want (%d,%d)\n",
                                 i, i / SIZE, i % SIZE, output[i].x, output[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelUpdateVertical: output row0 (got) ===\n  ");
    for (int c = 0; c < SIZE; ++c) printf("(%d,%d) ", output[c].x, output[c].y);
    printf("\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
