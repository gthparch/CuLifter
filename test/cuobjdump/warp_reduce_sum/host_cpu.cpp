#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z15warp_reduce_sumPKfPfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;
    constexpr int warpSize = 32;
    constexpr int warpCount = (gridDim * blockDim) / warpSize;

    std::vector<float> input(N, 0.0f);
    std::vector<float> warp_sums(warpCount, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }

    launchKernel(_Z15warp_reduce_sumPKfPfi, gridDim, blockDim,
                 input.data(), warp_sums.data(), N);

    bool ok = true;
    for (int warp = 0; warp < warpCount; ++warp) {
        const int start = warp * warpSize;
        const int end = std::min(N, start + warpSize);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += input[i];
        }
        if (std::fabs(warp_sums[warp] - expected) > 1e-5f) {
            std::cout << "FAIL at warp " << warp
                      << ": got " << warp_sums[warp]
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
