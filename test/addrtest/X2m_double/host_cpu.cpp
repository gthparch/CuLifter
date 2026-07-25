#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X2m_double();

int main() {
    constexpr int N = 1024, NPTRS = 2;

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

    int which = 0;
    Dim3 grid{(N + 255) / 256, 1};
    Dim3 block{256, 1};
    launchKernel2D(X2m_double, grid, block,
                   (const double* const*)ptrs, out, N, which);

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        double expected = bufs[which][i-1] + bufs[which][i] + bufs[which][i+1];
        if (std::fabs(out[i] - expected) > 1e-10) {
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
