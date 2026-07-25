#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void M2m_double();

int main() {
    constexpr int N = 1024, NPTRS = 3;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    double *bufs[NPTRS];
    for (int p = 0; p < NPTRS; p++) {
        bufs[p] = new double[N];
        for (int i = 0; i < N; i++)
            bufs[p][i] = (double)((i % 7 + 1 + p) * 0.1f);
    }

    const double* ptrs[NPTRS];
    for (int p = 0; p < NPTRS; p++) ptrs[p] = bufs[p];

    double *out = new double[N];
    std::memset(out, 0, N * sizeof(double));

    int which = 1;
    launchKernel(M2m_double, gridDim, blockDim,
                 (const double* const*)ptrs, out, N, which);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (out[i] != bufs[which][i]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << bufs[which][i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    for (int p = 0; p < NPTRS; p++) delete[] bufs[p];
    delete[] out;
    return ok ? 0 : 1;
}
