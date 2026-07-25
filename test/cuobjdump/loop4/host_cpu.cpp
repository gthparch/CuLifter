#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void _Z5loop4PKfS0_Pfi();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 128;
    constexpr int gridDim  = (N + blockDim - 1) / blockDim;
    const int totalThreads = gridDim * blockDim;

    float *A = new float[N];
    float *B = new float[N];
    float *C = new float[N];

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        A[i] = dist(gen);
        B[i] = dist(gen);
    }

    std::fill(C, C + N, 0.0f);

    launchKernel(_Z5loop4PKfS0_Pfi, gridDim, blockDim, A, B, C, N);

    bool ok = true;
    for (int tid = 0; tid < N; ++tid) {
        float expected = 0;
        for (int i = tid; i < N; i += 1)
            expected += A[i] + B[i];

        float tol = 1e-3 * std::fabs(expected);
        if (std::fabs(C[tid] - expected) > tol) {
            std::cerr << "Mismatch at " << tid << ": got " << C[tid]
                      << ", expected " << expected << "\n";
            ok = false;
            break;
        }
    }

    if (ok) std::cout << "TEST PASSED\n";

    delete[] A;
    delete[] B;
    delete[] C;
    return ok ? 0 : 1;
}
