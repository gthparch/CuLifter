#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z16warp_any_nonzeroPKiPii();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;
    constexpr int warpSize = 32;
    constexpr int warpCount = (gridDim * blockDim) / warpSize;

    std::vector<int> flags(N, 0);
    std::vector<int> warp_any(warpCount, 0);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(-2, 2);
    for (int i = 0; i < N; ++i) {
        flags[i] = dist(gen);
    }

    launchKernel(_Z16warp_any_nonzeroPKiPii, gridDim, blockDim,
                 flags.data(), warp_any.data(), N);

    bool ok = true;
    for (int warp = 0; warp < warpCount; ++warp) {
        const int start = warp * warpSize;
        const int end = std::min(N, start + warpSize);
        int expected = 0;
        for (int i = start; i < end; ++i) {
            if (flags[i] != 0) {
                expected = 1;
                break;
            }
        }
        if (warp_any[warp] != expected) {
            std::cout << "FAIL at warp " << warp
                      << ": got " << warp_any[warp]
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
