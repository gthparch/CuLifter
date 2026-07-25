#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z15warp_prefix_sumPKiPii();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;
    constexpr int warpSize = 32;
    constexpr int warpCount = (gridDim * blockDim + warpSize - 1) / warpSize;

    std::vector<int> input(N, 0);
    std::vector<int> output(N, 0);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<int> dist(-4, 4);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }

    launchKernel(_Z15warp_prefix_sumPKiPii, gridDim, blockDim,
                 input.data(), output.data(), N);

    bool ok = true;
    for (int warp = 0; warp < warpCount && ok; ++warp) {
        const int start = warp * warpSize;
        const int end = std::min(N, start + warpSize);
        int running = 0;
        for (int idx = start; idx < end; ++idx) {
            running += input[idx];
            if (output[idx] != running) {
                std::cout << "FAIL at index " << idx
                          << ": got " << output[idx]
                          << ", expected " << running << "\n";
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    return ok ? 0 : 1;
}
