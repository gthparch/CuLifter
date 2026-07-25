#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void WS1_warp_max();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[8];
    std::memset(out, 0, 8 * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)((i * 7 + 3) % 200);

    launchKernel(WS1_warp_max, gridDim, blockDim,
                 (const float*)in, out, N);

    bool ok = true;
    for (int w = 0; w < 8; w++) {
        float mx = -1e30f;
        for (int i = w * 32; i < (w + 1) * 32; i++) mx = fmaxf(mx, in[i]);
        if (fabsf(out[w] - mx) > 0.1f) {
            std::cout << "FAIL warp " << w << ": got " << out[w]
                      << " expected " << mx << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
