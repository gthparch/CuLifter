#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void N4_shifted_base();

int main() {
    constexpr int N = 1024;
    constexpr int shift = 10;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    launchKernel(N4_shifted_base, gridDim, blockDim,
                 (const float*)in, out, N, shift);

    bool ok = true;
    for (int i = 1; i < N - shift - 1; i++) {
        float e = in[shift + i - 1] + in[shift + i] + in[shift + i + 1];
        if (std::fabs(out[i] - e) > 1e-5f) {
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
