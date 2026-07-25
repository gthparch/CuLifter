#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS0_warp_sum();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[8];
    std::memset(out, 0, 8 * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i + 1);

    launchKernel(WS0_warp_sum, gridDim, blockDim,
                 (const float*)in, out, N);

    bool ok = true;
    for (int w = 0; w < 8; w++) {
        float exp = 0;
        for (int i = w * 32; i < (w + 1) * 32; i++) exp += in[i];
        if (fabsf(out[w] - exp) > 1.0f) {
            std::cout << "FAIL warp " << w << ": got " << out[w]
                      << " expected " << exp << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
