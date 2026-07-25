#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS4_prefix_sum();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = 1.0f;

    launchKernel(WS4_prefix_sum, gridDim, blockDim,
                 (const float*)in, out, N);

    // Inclusive prefix sum within each warp: out[w*32+i] = i+1
    bool ok = true;
    for (int w = 0; w < N / 32; w++) {
        for (int i = 0; i < 32; i++) {
            float expected = (float)(i + 1);
            if (fabsf(out[w * 32 + i] - expected) > 0.5f) {
                std::cout << "FAIL at " << (w * 32 + i) << ": got "
                          << out[w * 32 + i] << " expected " << expected << "\n";
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
