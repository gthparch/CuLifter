#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS2_warp_xor_sum();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i + 1);

    launchKernel(WS2_warp_xor_sum, gridDim, blockDim,
                 (const float*)in, out, N);

    // XOR butterfly reduction: every thread in warp gets the same warp sum
    bool ok = true;
    for (int w = 0; w < 8; w++) {
        float wsum = 0;
        for (int i = w * 32; i < (w + 1) * 32; i++) wsum += in[i];
        for (int i = w * 32; i < (w + 1) * 32; i++) {
            if (fabsf(out[i] - wsum) > 1.0f) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << wsum << "\n";
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
