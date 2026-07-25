// host_cpu.cpp — driver for warp_partialwarp_reduce.
//
// Reproduces the generic warp-reduction mass-drop bug (so.193 cuDNN softmax
// sum-stage): warp 0 lanes 0..7 run an EXTRA partial-warp block-reduce, then
// ALL lanes __syncthreads(), then a SECOND full-warp SHFL.DOWN reduce with
// NON-UNIFORM per-lane values. A runtime that models warp membership by a
// per-lane divergence/collective-count history splits warp0 into {0..7}/{8..31}
// after the barrier and DROPS mass from the second reduction. The block barrier
// is a full reconvergence point, so the second full-warp reduce must see all 32
// lanes (fixed by warp_hist_resync_after_barrier in kernel_wrapper_75.h).
//
// blocksum1 = sum(in)              -- reduced BEFORE the asymmetry bites (control)
// blocksum2 = sum(in*1.5 - 0.25*lane)  -- reduced AFTER barrier+partial collective
//
// sm_90 layout: args at c[0x0][0x210]; pinned explicitly via ArgAt so the
// default ARG_BASE does not matter.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void mb_partialwarp();

int main() {
    constexpr int N = 256;
    float* in  = new float[N];
    float* out = new float[8];
    for (int i = 0; i < N; i++) in[i] = (float)((i * 37 + 11) % 256 - 128) * 0.03125f;
    for (int i = 0; i < 8; i++) out[i] = -999.0f;

    launchKernel(mb_partialwarp, 1, N,
                 ArgAt{0x210}, in, ArgAt{0x218}, out, ArgAt{0x220}, (int)N);

    double ref1 = 0, ref2 = 0;
    for (int i = 0; i < N; i++) {
        ref1 += in[i];
        int lane = i & 31;
        ref2 += (double)in[i] * 1.5 - 0.25 * (double)lane;
    }
    double d1 = fabs((double)out[0] - ref1);
    double d2 = fabs((double)out[1] - ref2);
    printf("blocksum1 x86=%.6f ref=%.6f diff=%.3e\n", out[0], ref1, d1);
    printf("blocksum2 x86=%.6f ref=%.6f diff=%.3e\n", out[1], ref2, d2);
    bool ok = d1 < 1e-2 && d2 < 1e-2;
    printf("%s\n", ok ? "TEST PASSED" : "DIVERGED (post-barrier full-warp reduce dropped mass)");
    return 0;
}
