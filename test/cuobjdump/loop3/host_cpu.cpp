#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include "kernel_wrapper.h"

extern "C" void _Z5loop3PKfS0_Pfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;

    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];


    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for(int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
        C[i] = 0.0f;
    }

    launchKernel(_Z5loop3PKfS0_Pfi, gridDim, blockDim,
                          A, B, C, N);

    bool ok = true;
    for (int idx = 0; idx < gridDim * blockDim; ++idx) {
        double expected = 0.0;
        for (int i = idx; i < N; i += gridDim * blockDim)
            expected += static_cast<double>(A[i]+B[i]);

        double diff = std::fabs(static_cast<double>(C[idx]) - expected);
        double tol  = 1e-3 * std::max(1.0, std::fabs(expected));
        if (diff > tol) {
            std::cerr << "Mismatch at thread " << idx << ": got " << C[idx]
                      << ", expected ~" << expected << std::endl;
            ok = false;
            break;
        }
    }

    if (ok) {
        std::cout << "TEST PASSED\n";
    }

    delete[] A;
    delete[] B;
    delete[] C;
    return ok ? 0 : 1;
}
