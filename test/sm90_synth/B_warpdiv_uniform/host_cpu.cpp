#include <cstdio>
#include "kernel_wrapper.h"
extern "C" void kern_warpdiv();
int main() {
    // blockDim 80 (not a multiple of 32): warpid 0..2, nwarps = 80/32 = 2,
    // lane = t%32 — exercises both the vector IMAD.HI leaf (tid/32) and the
    // uniform UIMAD.WIDE.U32 leaf (blockDim/32) plus their corrections.
    constexpr int GRID = 2, BLOCK = 80, N = GRID * BLOCK;
    int* q = new int[1024]();
    int* w = new int[1024]();
    int* l = new int[1024]();
    launchKernel(kern_warpdiv, GRID, BLOCK, ArgAt{0x210}, q, ArgAt{0x218}, w,
                 ArgAt{0x220}, l, ArgAt{0x228}, N);
    int bad = 0;
    for (int i = 0; i < N; i++) {
        int t = i % BLOCK;
        int ew = t / 32, en = BLOCK / 32, el = t % 32;
        bool ok = (q[i] == ew) && (w[i] == en) && (l[i] == el);
        if (!ok) {
            bad++;
            if (bad <= 5)
                printf("%3d: lifted (w=%d n=%d l=%d), native (w=%d n=%d l=%d) MISMATCH\n",
                       i, q[i], w[i], l[i], ew, en, el);
        }
    }
    printf("bad=%d\n", bad);
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
