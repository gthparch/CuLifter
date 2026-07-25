#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void H4m_large();

int main() {
    constexpr int W = 64, H = 64, total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)((i % 7 + 1) * 0.1f);

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(H4m_large, grid, block,
                   (const float*)in, out, (size_t)W, W, H);

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
