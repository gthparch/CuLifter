#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_N1_stencil_3pt_double();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    double *in = new double[N];
    double *out = new double[N];
    std::memset(out, 0, N * sizeof(double));
    for (int i = 0; i < N; i++) in[i] = (double)(i % 7 + 1) * 0.1;

    launchKernel(addr_N1_stencil_3pt_double, gridDim, blockDim,
                 (const double*)in, out, N);

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        double expected = in[i - 1] + in[i] + in[i + 1];
        if (std::fabs(out[i] - expected) > 1e-10) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
