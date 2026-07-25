#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include "kernel_wrapper.h"

extern "C" void vector_add();

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
    }

    launchKernel(vector_add, gridDim, blockDim,
                          A, B, C, N);

    bool ok = true;
    for(int i = 0; i < N; ++i) {
        float expected = A[i] + B[i];
        if (std::fabs(C[i] - expected) > 1e-6f) {
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

    delete[] A;
    delete[] B;
    delete[] C;
    return ok ? 0 : 1;
}
