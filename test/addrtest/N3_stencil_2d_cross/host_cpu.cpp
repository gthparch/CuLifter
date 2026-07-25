#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void N3_stencil_2d_cross();

int main() {
    constexpr int W = 32, H = 32, total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(N3_stencil_2d_cross, grid, block,
                   (const float*)in, out, W, H);

    bool ok = true;
    for (int r = 1; r < H - 1; r++) {
        for (int c = 1; c < W - 1; c++) {
            int i = r * W + c;
            float e = in[i - W] + in[i + W] + in[i - 1] + in[i + 1] + in[i];
            if (std::fabs(out[i] - e) > 1e-5f) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << e << "\n";
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
