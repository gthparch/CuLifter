#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
// Runs the LIFTED cross-warp producer/consumer mbarrier kernel. Correct only if
// the consumer warp actually blocks on the producer warp's mbarrier arrivals
// (a real mbarrier); the always-ready NOP model races and reads 0.
extern "C" void mbar_prod_cons();

int main() {
    const int N = 32;
    float in[N], out[N];
    for (int i = 0; i < N; i++) { in[i] = (float)(i + 1); out[i] = -1.0f; }

    launchKernel(mbar_prod_cons, 1, 64, in, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = 2.0f * (float)(i + 1);
        if (fabsf(out[i] - exp) > 1e-5f) {
            printf("FAIL at %d: got %f exp %f\n", i, out[i], exp);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
