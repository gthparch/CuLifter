#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void H4m_double();

int main() {
    constexpr int W = 32, H = 32, total = W * H;

    double *in = new double[total];
    double *out = new double[total];
    std::memset(out, 0, total * sizeof(double));
    for (int i = 0; i < total; i++) in[i] = (double)((i % 7 + 1) * 0.1f);

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(H4m_double, grid, block,
                   (const double*)in, out, (size_t)W, W, H);

    bool ok = true;
    for (int i = 0; i < total; i++) {
        if (out[i] != in[i]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << in[i] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
