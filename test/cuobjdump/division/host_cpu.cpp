#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "kernel_wrapper.h"

extern "C" void vector_div();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    std::vector<float> A(N, 0.0f);
    std::vector<float> B(N, 0.0f);
    std::vector<float> C(N, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> a_dist(-10.0f, 10.0f);
    std::uniform_int_distribution<int> denom_dist(1, 10);
    std::bernoulli_distribution sign_dist(0.5);
    for (int i = 0; i < N; ++i) {
        A[i] = a_dist(gen);
        int denom = denom_dist(gen);
        if (sign_dist(gen)) {
            denom = -denom;
        }
        B[i] = static_cast<float>(denom);
    }

    launchKernel(vector_div, gridDim, blockDim,
                 A.data(), B.data(), C.data(), N);

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float expected = A[i] / B[i];
        if (std::fabs(C[i] - expected) > 1e-4f) {
            std::cout << "FAIL at index " << i
                      << ": got " << C[i]
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
