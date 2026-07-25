#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_flat2d_f32();

int main() {
    constexpr int W = 32, H = 32, pitch = W;
    float *h_in = new float[W*H];
    float *h_out = new float[W*H];
    memset(h_out, 0, W*H*sizeof(float));
    for (int i = 0; i < W*H; i++) h_in[i] = (float)((i%7)+1)*0.1f;

    launchKernel2D(S_flat2d_f32, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   (float*)h_in, h_out, W, H, pitch);

    bool ok = true;
    for (int i = 0; i < W*H; i++) {
        if (h_out[i] != h_in[i]) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
