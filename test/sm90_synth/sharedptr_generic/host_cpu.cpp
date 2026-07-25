#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void kern_sharedptr_generic();

int main() {
    constexpr int N = 64;
    float* in = new float[N];
    float* out = new float[N];
    for (int i = 0; i < N; i++) { in[i] = (float)(i + 1); out[i] = 0.0f; }

    launchKernel(kern_sharedptr_generic, 1, N, out, (const float*)in, N, 1);

    // flip=1: s=b (2x), sn=a (1x) -> out = 2*in + 0.5*in = 2.5*in
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(out[i] - 2.5f * in[i]) > 1e-5f) {
            printf("MISMATCH i=%d got %f want %f\n", i, out[i], 2.5f * in[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
