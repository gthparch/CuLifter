#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void A4_axpy();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;
    float a = 2.0f, b = 0.5f;

    launchKernel(A4_axpy, gridDim, blockDim,
                 (const float*)in, out, N, a, b);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = a * in[i] + b;
        if (std::fabs(out[i] - expected) > 1e-5f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
