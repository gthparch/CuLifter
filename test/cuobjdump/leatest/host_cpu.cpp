#include <cstdint>
#include <iostream>
#include <cmath>
#include <random>
#include "kernel_wrapper.h"

extern "C" void _Z27minimal_address_calculationPfi();

int main() {
    constexpr int N = 128;
    constexpr int blockDim = 32;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    // Allocate memory for a simple 1D array test first
    float *C = new float[N];
    
    // Initialize with known values
    for(int i = 0; i < N; ++i) {
        C[i] = -1.0f;  // Initialize to a known value
    }

    // Use simple 1D grid for now to avoid segfault
    launchKernel(_Z27minimal_address_calculationPfi, gridDim, blockDim,
                          C, N);

    bool ok = true;
    // Check results - each thread should store its thread index as a float
    for(int i = 0; i < N; ++i) {
        int blockId = i / blockDim;
        int threadId = i % blockDim;
        int globalId = blockId * blockDim + threadId;
        
        if (globalId < N) {
            float expected = (float)globalId;
            float actual = C[i];
            
            if (std::fabs(actual - expected) > 1e-6f) {
                std::cout << "FAIL at index " << i
                          << ": got " << actual
                          << ", expected " << expected << std::endl;
                ok = false;
                break;
            }
        }
    }

    if (ok) {
        std::cout << "TEST PASSED" << std::endl;
    }

    delete[] C;
    return ok ? 0 : 1;
}
