#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_X2_ptr_stencil_float();

int main() {
    constexpr int N = 1024;
    constexpr int NPTRS = 2;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        bufs[p] = new float[N];
        for (int i = 0; i < N; i++)
            bufs[p][i] = (float)((i % 7 + 1 + p) * 0.1f);
    }

    const float* ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) ptrs[p] = bufs[p];

    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));

    int which = 0;
    launchKernel(addr_X2_ptr_stencil_float, gridDim, blockDim,
                 (const float* const*)ptrs, out, N, which);

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        float expected = bufs[which][i-1] + bufs[which][i] + bufs[which][i+1];
        if (std::fabs(out[i] - expected) > 1e-5f) {
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
