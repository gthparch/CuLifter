// x86 driver for kern_cluster_ctarank (SM90 SR_CgaCtaId microbench).
//
// Two blocks of 64 threads (2 warps — exercises BOTH the uniform S2UR
// SR_CgaCtaId read and the @!P-predicated per-thread S2R reads, plus the
// warp==0 second-stage reduce whose divergent vote.sync shape crashed
// ptxas on the const-0 lift). Each block sums the same 96 floats.
//
// Cluster semantics on x86: the wrapper models cluster dims (1,1,1), so
// SR_CgaCtaId == 0 is the EXACT runtime value (the kernel is launched
// without clusters on hardware too); the (rank<<24)+0x400 shared::cluster
// addressing then stays inside the CTA's own shared_mem buffer.
//
// Values are small integers, so every fp32 partial sum is exact and the
// expected total is order-independent.
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_cluster_ctarank();

int main() {
    constexpr int N = 96;
    constexpr int BLOCKS = 2;
    float* in = new float[N];
    float* out = new float[BLOCKS]();
    float want = 0.0f;
    for (int i = 0; i < N; i++) {
        in[i] = float(i % 7 + 1);
        want += in[i];
    }
    launchKernel(kern_cluster_ctarank, BLOCKS, 64, out, in, N);
    int bad = 0;
    for (int b = 0; b < BLOCKS; b++) {
        bool ok = (out[b] == want);
        if (!ok) bad++;
        printf("block %d: lifted %.1f, native %.1f  %s\n", b, out[b], want,
               ok ? "ok" : "MISMATCH");
    }
    if (bad == 0) printf("TEST PASSED\n");
    return bad ? 1 : 0;
}
