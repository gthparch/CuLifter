#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P1_many_scalars();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    float a0 = 1.0f, a1 = 0.1f, a2 = 0.2f, a3 = 0.3f;
    float a4 = 0.4f, a5 = 0.5f, a6 = 0.6f, a7 = 0.7f;

    launchKernel(P1_many_scalars, gridDim, blockDim,
                 (const float*)in, out, N, a0, a1, a2, a3, a4, a5, a6, a7);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float e = in[i] * a0 + a1 + a2 - a3 + a4 * a5 + a6 - a7;
        if (std::fabs(out[i] - e) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << e << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
