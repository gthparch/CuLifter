#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X1m_radius2();

int main() {
    constexpr int W = 32, H = 32, total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)((i % 7 + 1) * 0.1f);

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(X1m_radius2, grid, block,
                   (const float*)in, out, (size_t)W, W, H);

    bool ok = true;
    for (int row = 2; row < H - 2; row++) {
        for (int col = 2; col < W - 2; col++) {
            int c = row * W + col;
            float expected = in[(row-2)*W + col] + in[row*W + (col-2)]
                           + in[(row+2)*W + col] + in[row*W + (col+2)]
                           + in[row*W + col];
            if (std::fabs(out[c] - expected) > 1e-5f) {
                std::cout << "FAIL at row=" << row << " col=" << col
                          << " got=" << out[c] << " expected=" << expected << "\n";
                ok = false;
                goto done;
            }
        }
    }
    done:
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
