#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS3_broadcast_lane0();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i * 7 + 3);

    launchKernel(WS3_broadcast_lane0, gridDim, blockDim,
                 (const float*)in, out, N);

    // Broadcast: every thread in warp gets lane 0's value
    bool ok = true;
    for (int w = 0; w < N / 32; w++) {
        float expected = in[w * 32]; // lane 0's value
        for (int i = w * 32; i < (w + 1) * 32; i++) {
            if (out[i] != expected) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << expected << "\n";
                ok = false;
                goto done;
            }
        }
    }
done:
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
