#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_M2_ptr_array_float();

int main() {
    constexpr int N = 1024;
    constexpr int NPTRS = 3;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    // Allocate source buffers
    float *bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        bufs[p] = new float[N];
        for (int i = 0; i < N; i++)
            bufs[p][i] = (float)((i % 7 + 1 + p) * 0.1f);
    }

    // Build pointer array (array of const float*)
    const float* ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) ptrs[p] = bufs[p];

    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    int which = 1;
    launchKernel(addr_M2_ptr_array_float, gridDim, blockDim,
                 (const float* const*)ptrs, out, N, which);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = bufs[which][i];
        if (out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    for (int p = 0; p < NPTRS; p++) delete[] bufs[p];
    delete[] out;
    return ok ? 0 : 1;
}
