#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z19tile4_reduce_kernelPKfPfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;
    constexpr int tileSize = 4;
    constexpr int tileCount = (N + tileSize - 1) / tileSize;

    std::vector<float> input(N);
    std::vector<float> output(tileCount, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        input[i] = dist(gen);
    }

    launchKernel(_Z19tile4_reduce_kernelPKfPfi, gridDim, blockDim,
                 input.data(), output.data(), N);

    bool ok = true;
    for (int tile = 0; tile < tileCount; ++tile) {
        const int start = tile * tileSize;
        const int end = std::min(N, start + tileSize);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += input[i];
        }

        if (std::fabs(output[tile] - expected) > 1e-5f) {
            std::cerr << "Mismatch at tile " << tile << ": got " << output[tile]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) std::cout << "TEST PASSED\n";

    return ok ? 0 : 1;
}
