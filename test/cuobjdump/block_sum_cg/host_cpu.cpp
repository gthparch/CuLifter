#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z12block_sum_cgPKfPfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    std::vector<float> input(N, 0.0f);
    std::vector<float> block_sums(gridDim, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }

    launchKernel(_Z12block_sum_cgPKfPfi, gridDim, blockDim,
                 input.data(), block_sums.data(), N);

    bool ok = true;
    for (int block = 0; block < gridDim; ++block) {
        const int start = block * blockDim;
        const int end = std::min(N, start + blockDim);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += input[i];
        }
        if (std::fabs(block_sums[block] - expected) > 1e-5f) {
            std::cout << "FAIL at block " << block
                      << ": got " << block_sums[block]
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
