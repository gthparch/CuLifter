#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void D2_elem_double();

int main() {
    constexpr int W = 32, H = 32;

    double *h_in = new double[W*H];
    double *h_out = new double[W*H];
    memset(h_out, 0, W*H*sizeof(double));
    for (int i = 0; i < W*H; i++) h_in[i] = (double)((i % 7 + 1) * 0.1f);
    launchKernel2D(D2_elem_double, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   h_in, h_out, W, H);

    bool ok = true;
    for (int row = 1; row < H-1; row++) {
        for (int col = 0; col < W; col++) {
            int c = row*W+col;
            double expected = h_in[c-W] + h_in[c] + h_in[c+W];
            if (fabs(h_out[c] - expected) > 1e-10) {
                std::cout << "FAIL at row=" << row << " col=" << col << std::endl;
                ok = false; goto done;
            }
        }
    }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
