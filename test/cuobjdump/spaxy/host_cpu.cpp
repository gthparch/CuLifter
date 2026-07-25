#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void _Z5saxpyifPKfPf();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    float a = 2.0f;
    std::vector<float> x(N, 0.0f);
    std::vector<float> y(N, 0.0f);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(int i = 0; i < N; ++i) {
        x[i] = dist(gen);
        y[i] = dist(gen);
    }
    
    auto result_y = y;

    launchKernel(_Z5saxpyifPKfPf, gridDim, blockDim,
                          N, a, x.data(), result_y.data());

    bool ok = true;
    for(int i = 0; i < N; ++i) {
        float expected = a * x[i] + y[i];
        if (std::fabs(result_y[i] - expected) > 1e-6f) {
            std::cout << "FAIL at index " << i
                      << ": got " << y[i]
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
