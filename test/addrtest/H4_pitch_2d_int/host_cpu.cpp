#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_H4_pitch_2d_int();

int main() {
    constexpr int W = 32, H = 32;
    constexpr int pitch = W;
    constexpr int total = W * H;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    Dim3 grid{(W + 15) / 16, (H + 15) / 16};
    Dim3 block{16, 16};
    launchKernel2D(addr_H4_pitch_2d_int, grid, block,
                   (const float*)in, out, pitch, W, H);

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
