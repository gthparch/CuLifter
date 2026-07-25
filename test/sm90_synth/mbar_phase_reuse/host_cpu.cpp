#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
// Runs the LIFTED mbarrier-phase-reuse kernel (loop reusing one mbarrier) and
// checks the cross-lane accumulation. Correct only if the real mbarrier orders
// each iteration's writes before the neighbor reads AND reuses the phase parity.
extern "C" void mbar_phase_reuse();

#define K 8

int main() {
    const int N = 32;
    float in[N], out[N];
    for (int i = 0; i < N; i++) { in[i] = (float)(i + 1); out[i] = -1.0f; }

    launchKernel(mbar_phase_reuse, 1, 32, in, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = 0.0f;
        for (int k = 0; k < K; k++) exp += in[(i + 1) & 31] + (float)k;
        if (fabsf(out[i] - exp) > 1e-4f) {
            printf("FAIL at %d: got %f exp %f\n", i, out[i], exp);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
