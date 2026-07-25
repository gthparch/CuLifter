#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void _Z11dot_productPKfS0_Pfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    std::vector<float> A(N, 0.0f);
    std::vector<float> B(N, 0.0f);
    std::vector<float> block_sums(gridDim, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    launchKernel(_Z11dot_productPKfS0_Pfi, gridDim, blockDim,
                 A.data(), B.data(), block_sums.data(), N);

    bool ok = true;
    for (int block = 0; block < gridDim; ++block) {
        const int start = block * blockDim;
        const int end = std::min(N, start + blockDim);
        float expected = 0.0f;
        for (int i = start; i < end; ++i) {
            expected += A[i] * B[i];
        }
        if (std::fabs(block_sums[block] - expected) > 1e-4f) {
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
