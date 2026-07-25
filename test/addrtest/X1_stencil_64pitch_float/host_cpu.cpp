#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_X1_stencil_64pitch_float();

int main() {
    constexpr int W = 32, H = 32;
    constexpr size_t pitch = W;
    constexpr int total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(addr_X1_stencil_64pitch_float, grid, block,
                   (const float*)in, out, pitch, W, H);

    bool ok = true;
    for (int row = 1; row < H - 1; row++) {
        for (int col = 1; col < W - 1; col++) {
            int c = row * W + col;
            int n = (row-1) * W + col;
            int s = (row+1) * W + col;
            int w = row * W + (col-1);
            int e = row * W + (col+1);
            float expected = in[n] + in[s] + in[w] + in[e] + in[c];
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
