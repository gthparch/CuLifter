#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

// Microbench: sm90 mbarrier arrive-on-cp.async
//   ARRIVES.LDGSTSBAR.64.TRANSCNT [mbar]  (cp.async.mbarrier.arrive)
// paired with cp.async (LDGSTS) global->shared and a try_wait consumer spin —
// the FlashAttention-forward producer handshake. On x86 the cp.async completes
// synchronously and the SYNCS.PHASECHK.TRYWAIT wait is always-ready, so the
// arrive that increments the count it would check is a NOP.
// Oracle: cp.async copies in[i] -> smem[i], so out[i] == in[i] + 1.0f.

extern "C" void mbarrier_arrive();

int main() {
    constexpr int N = 256;
    static float h_in[N], h_out[N];
    std::memset(h_out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    launchKernel(mbarrier_arrive, 1, N, h_in, h_out, N);

    int bad = 0;
    for (int i = 0; i < N; i++) {
        float ref = h_in[i] + 1.0f;
        if (std::fabs(h_out[i] - ref) > 1e-4f) {
            if (bad < 3)
                std::printf("MISMATCH i=%d got=%g ref=%g\n", i, h_out[i], ref);
            bad++;
        }
    }
    if (bad == 0) std::cout << "TEST PASSED" << std::endl;
    else          std::cout << "FAIL " << bad << std::endl;
    return bad != 0;
}
