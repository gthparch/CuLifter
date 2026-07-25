#include <cstdio>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_wideleaf();

// The 45_dual_gemm grid-strided WIDE-leaf shape:
//   out[clamp((i/N)*stride + i%N)] = in[i].
// A CORRECT unsigned divide makes index<->off a bijection over a [rows x stride]
// region; an off-by quotient (the unrecognized magic chain lifted faithfully
// via the software rcp_f32) scatters to wrong addresses => 'bad' mismatches at
// the expected cells (and the buggy quotient can be wildly off — the clamp in
// the kernel keeps every store in-bounds so the bug shows as value mismatches,
// not a segfault). N is NON-pow2 (the magic-rcp path; a pow-2 N lowers to a
// shift and dodges the bug). stride >= N so distinct (coord0,coord1) never alias.
static int run(unsigned N, long stride, unsigned total) {
    unsigned rows = (total + N - 1) / N;                 // ceil
    long region = (long)rows * stride;                   // exact footprint
    long cap = region;                                   // clamp bound = region
    std::vector<int> out(region, -1), in(total);
    for (unsigned i = 0; i < total; i++) in[i] = (int)(i * 2654435761u ^ 0x9e37u);

    off = ARG_BASE;
    launchKernel(kern_wideleaf, /*grid*/ (int)((total + 127) / 128), /*block*/ 128,
                 ArgAt{0x210}, out.data(), ArgAt{0x218}, in.data(),
                 ArgAt{0x220}, N, ArgAt{0x228}, stride, ArgAt{0x230}, total,
                 ArgAt{0x238}, cap);

    int bad = 0, unwritten = 0;
    for (unsigned i = 0; i < total; i++) {
        unsigned c0 = i / N, c1 = i % N;
        long o = (long)c0 * stride + c1;
        if (out[o] != in[i]) {
            if (out[o] == -1) unwritten++;
            if (++bad <= 6)
                printf("  i=%u expect out[%ld]=%d got %d (c0=%u c1=%u)\n",
                       i, o, in[i], out[o], c0, c1);
        }
    }
    printf("[N=%u stride=%ld total=%u rows=%u] bad=%d unwritten=%d\n",
           N, stride, total, rows, bad, unwritten);
    return bad;
}

int main() {
    int bad = 0;
    bad += run(/*N*/ 300, /*stride*/ 320, /*total*/ 4096);  // non-pow2 N (dual-gemm)
    bad += run(/*N*/ 100, /*stride*/ 100, /*total*/ 2000);  // stride==N tight pack
    bad += run(/*N*/ 7,   /*stride*/ 16,  /*total*/ 1000);  // tiny non-pow2 divisor
    printf("TOTAL bad=%d\n", bad);
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
