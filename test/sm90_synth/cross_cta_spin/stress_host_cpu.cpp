#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "kernel_wrapper.h"
extern "C" void streamk_stress();
static bool run1(int NPROD, int BLK) {
    const int GRID = NPROD + 1;
    const float base = 2.0f, contrib = 0.5f;
    const int nprod = NPROD * BLK;                 // owner waits for every producer THREAD
    const float expect = (float)nprod * contrib + base;
    float* out = new float[1]{-7.0f}; float* partials = new float[1]{0.0f}; int* done = new int[1]{0};
    launchKernel(streamk_stress, GRID, BLK,
        ArgAt{0x208},(uint64_t)0, ArgAt{0x210},out, ArgAt{0x218},partials,
        ArgAt{0x220},done, ArgAt{0x228},base, ArgAt{0x22c},contrib, ArgAt{0x230},(int)nprod);
    bool ok = std::fabs(out[0]-expect) < 1e-4f;
    printf("  grid=%d blk=%d nprod=%d: out=%.4f expect=%.4f partials=%.4f done=%d -> %s\n",
           GRID, BLK, nprod, out[0], expect, partials[0], done[0], ok?"OK":"WRONG");
    delete[] out; delete[] partials; delete[] done;
    return ok;
}
int main() {
    bool a = run1(4, 1);   // 4 producer CTAs, 1 thread each
    bool b = run1(1, 4);   // 1 producer CTA, 4 threads
    bool c = run1(4, 4);   // 4 producer CTAs, 4 threads each
    bool ok = a && b && c;
    printf("%s\n", ok ? "TEST PASSED" : "DIVERGED");
    fflush(stdout); _Exit(ok?0:1);
}
