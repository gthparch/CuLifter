#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void A5_two_input();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in1 = new float[N];
    float *in2 = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        in1[i] = (float)(i % 7 + 1) * 0.1f;
        in2[i] = (float)(i % 5 + 1) * 0.2f;
    }

    launchKernel(A5_two_input, gridDim, blockDim,
                 (const float*)in1, (const float*)in2, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = in1[i] + in2[i];
        if (std::fabs(out[i] - expected) > 1e-5f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in1; delete[] in2; delete[] out;
    return ok ? 0 : 1;
}
