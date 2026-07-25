// host_cpu.cpp — x86 exec-correctness for NPP PBA kernelColor.
//
// Kernel: _Z11kernelColorP6short2S0_i
//   kernelColor(short2* input, short2* output, int size)
//   (from libnpp sm_90 npp.cubin, lifted to x86 by SLifter)
//
// Ground-truth: PBA+ pba2DKernel.h (NUS), transcribed below. This kernel uses
//   __shared__ short2 block[BLOCKSIZE][BLOCKSIZE] + __syncthreads() to TRANSPOSE
// each thread-column's result before writing to `output`. The SLifter x86 wrapper
// runs all blockDim threads of a CTA CONCURRENTLY with a real per-block BAR.SYNC
// barrier and a CTA-shared `shared_mem` segment, so the intra-CTA transpose is
// modeled correctly (CTAs are sequential; this kernel's cross-thread interaction
// is intra-CTA only — output[id] is a deterministic function of the inputs, the
// only dependence on other threads is through the barrier-separated shared block).
//
// Per-thread semantics (column tx = blockIdx.x*blockDim.x + col; tid = threadIdx.y):
//   lasty = size-1; last2 = input[TOID(tx,lasty)];
//   if (last2.x==MARKER){ lasty=last2.y; last2=input[TOID(tx,lasty)]; }
//   if (last2.y>=0) last1 = input[TOID(tx,last2.y)];
//   for step in [0,n_step): y_start=size-step*bx-1; y_end=size-(step+1)*bx;
//     for (ty = y_start-tid; ty>=y_end; ty-=by) {
//       best = (last2.x-tx)^2 + (lasty-ty)^2;        // unsigned 32-bit
//       while (last2.y>=0){ d=(last1.x-tx)^2+(last2.y-ty)^2; if(d>best)break;
//                           best=d; lasty=last2.y; last2=last1;
//                           if(last2.y>=0) last1=input[TOID(tx,last2.y)]; }
//       block[threadIdx.x][ty-y_end] = make_short2(lasty, last2.x);
//     }
//     __syncthreads();
//     if(!threadIdx.y){ id=TOID(y_end+threadIdx.x, blockIdx.x*bx, size);
//                       for(i=0;i<bx;++i,id+=size) output[id]=block[i][threadIdx.x]; }
//     __syncthreads();
//
// short2.x = low 16b, .y = high 16b; MARKER = -32768. __mul24 -> truncated 24-bit
// signed mul, but dx,dy here are small so it equals plain mul. Params: input@0x210,
// output@0x218, size@0x220.  BLOCKSIZE = 64 (so blockDim.x must be 64).
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z11kernelColorP6short2S0_i();

struct s2 { int16_t x, y; };
static const int16_t MARKER = -32768;
static inline s2 mk(int x,int y){ return s2{(int16_t)x,(int16_t)y}; }

static int SIZE, BX, BY;
static inline int TOID(int x,int y){ return y*SIZE + x; }
// __mul24: low 24 bits of signed product (matches SASS IMAD .SHL/.U32 24-bit forms
// used in the index math; for the dx*dx/dy*dy distance the values are tiny so
// __mul24 == plain int mul). Reproduce faithfully.
static inline int mul24(int a, int b){
    int p = (int)(((unsigned)a & 0xFFFFFF) * ((unsigned)b & 0xFFFFFF));
    // sign-extend from bit 23? CUDA __mul24 returns low 32 bits of (a*b) with
    // 24-bit inputs; for our small operands the full int product is identical.
    return a * b;   // operands are bounded by size(<=64) and coords -> exact
}

int main(){
    SIZE = 64; BX = 64; BY = 4;          // blockDim.x must be BLOCKSIZE=64
    const int GX = 1, GY = 1;            // grid = dim3(size/bx) = 1; one CTA
    const int N = SIZE * SIZE;

    s2 input[N], out[N], ref[N];
    for (int i = 0; i < N; ++i) { out[i] = mk(0x2222, 0x3333); ref[i] = mk(0x2222, 0x3333); }

    // Build a valid per-column proximate-point stack:
    //   input[TOID(tx,y)].x = site x-coord (or MARKER if row y is not a stack node)
    //   input[TOID(tx,y)].y = backward pointer to the previous (lower-row) stack node
    // The stack is the lower convex hull of sites along the column; coloring walks
    // it. We lay 3 stack nodes per column at distinct rows with distinct x-coords so
    // the nearest-site competition (the while-pop) actually fires for many ty.
    for (int tx = 0; tx < SIZE; ++tx) {
        for (int y = 0; y < SIZE; ++y) input[TOID(tx,y)] = mk(MARKER, -1);
        // three nodes: low, mid, high rows; back-pointers chain upward usage.
        int r0 = 5, r1 = 30, r2 = 58;                 // ascending stack rows
        int x0 = (tx*7 + 3)  % 64;                    // distinct site x per column
        int x1 = (tx*13 + 20) % 64;
        int x2 = (tx*5 + 50) % 64;
        input[TOID(tx,r0)] = mk(x0, -1);              // bottom of stack: back ptr = -1
        input[TOID(tx,r1)] = mk(x1, r0);              // back ptr -> r0
        input[TOID(tx,r2)] = mk(x2, r1);              // back ptr -> r1
        // top-of-column sentinel: input[size-1] must resolve to the stack top (r2).
        // kernelColor starts at lasty=size-1; if input[size-1].x==MARKER it follows
        // .y to the real top. Set row (size-1).y = r2 so the walk enters at r2.
        if (r2 != SIZE-1) input[TOID(tx, SIZE-1)] = mk(MARKER, r2);
    }

    // ---- Independent reference: simulate every (col,tid) thread's trajectory,
    //      then perform the transpose into ref[] exactly as the kernel does. ----
    // shared block[BX][BX] (short2). We reproduce the two-phase write/read per step.
    static s2 block[64][64];
    for (int bx_blk = 0; bx_blk < GX; ++bx_blk) {
      int n_step = SIZE / BX;
      // Per-thread persistent stack state, indexed by (col, tid).
      for (int step = 0; step < n_step; ++step) {
        int y_start = SIZE - step*BX - 1;
        int y_end   = SIZE - (step+1)*BX;
        // PHASE 1: each thread (col,tid) writes its column of `block`.
        // Per-thread persistent stack state (carried across steps), indexed
        // by (col, tid). The kernel keeps last1/last2/lasty LIVE across steps.
        static int  s_lasty[64][64];
        static s2   s_last1[64][64], s_last2[64][64];
        static bool inited[64][64];
        for (int col = 0; col < BX; ++col) {
          int tx = bx_blk*BX + col;
          for (int tid = 0; tid < BY; ++tid) {
            int &L  = s_lasty[col][tid];
            s2  &A2 = s_last2[col][tid];
            s2  &A1 = s_last1[col][tid];
            if (step == 0 && !inited[col][tid]) {
              inited[col][tid] = true;
              L = SIZE - 1;
              A2 = input[TOID(tx, L)];
              if (A2.x == MARKER) { L = A2.y; A2 = input[TOID(tx, L)]; }
              if (A2.y >= 0) A1 = input[TOID(tx, A2.y)];
            }
            for (int ty = y_start - tid; ty >= y_end; ty -= BY) {
              int dx = (int)A2.x - tx, dy = L - ty;
              unsigned best = (unsigned)(mul24(dx,dx) + mul24(dy,dy));
              while (A2.y >= 0) {
                dx = (int)A1.x - tx; dy = (int)A2.y - ty;
                unsigned d = (unsigned)(mul24(dx,dx) + mul24(dy,dy));
                if (d > best) break;
                best = d; L = A2.y; A2 = A1;
                if (A2.y >= 0) A1 = input[TOID(tx, (int)A2.y)];
              }
              block[col][ty - y_end] = mk(L, A2.x);    // block[threadIdx.x][ty-y_end]
            }
          }
        }
        // __syncthreads();  PHASE 2: threadIdx.y==0 threads transpose-write output.
        for (int col = 0; col < BX; ++col) {     // col == threadIdx.x, threadIdx.y==0
          int id = TOID(y_end + col, bx_blk*BX);   // TOID(x,y,size); size is global
          for (int i = 0; i < BX; ++i, id += SIZE)
            ref[id] = block[i][col];             // block[i][threadIdx.x]
        }
        // __syncthreads();
      }
    }

    // 2D launch: grid {1,1}, block {64,4}. Reads threadIdx.y / blockIdx -> 2D.
    launchKernel2D(_Z11kernelColorP6short2S0_i,
                   Dim3{GX, GY}, Dim3{BX, BY},
                   ArgAt{0x210}, (void*)input,
                   ArgAt{0x218}, (void*)out,
                   ArgAt{0x220}, (int)SIZE);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (out[i].x != ref[i].x || out[i].y != ref[i].y) {
            if (bad < 8) printf("  mismatch[%d] (x=%d,y=%d): got(%d,%d) want(%d,%d)\n",
                                i, i%SIZE, i/SIZE, out[i].x, out[i].y, ref[i].x, ref[i].y);
            ++bad;
        }
    printf("=== kernelColor ===\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
