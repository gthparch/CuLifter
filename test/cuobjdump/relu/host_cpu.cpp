#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void _Z4reluPKfPfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    std::vector<float> input(N, 0.0f);
    std::vector<float> output(N, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for(int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }

    launchKernel(_Z4reluPKfPfi, gridDim, blockDim,
                          input.data(), output.data(), N);

    bool ok = true;
    for(int i = 0; i < N; ++i) {
        float expected = std::max(0.0f, input[i]);
        if (std::fabs(output[i] - expected) > 1e-6f) {
            std::cout << "FAIL at index " << i
                      << ": got " << output[i]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    return ok ? 0 : 1;
}
