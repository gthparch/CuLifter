#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_copy_u8_2d();

int main() {
    constexpr int W = 32, H = 32;
    uint8_t *h_in = new uint8_t[W*H];
    uint8_t *h_out = new uint8_t[W*H];
    memset(h_out, 0, W*H*sizeof(uint8_t));
    for (int i = 0; i < W*H; i++) h_in[i] = (uint8_t)((i%7)+1);

    launchKernel2D(S_copy_u8_2d, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   (uint8_t*)h_in, h_out, W, H);

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
