#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_sten3_u16_2d();

int main() {
    constexpr int W = 32, H = 32;
    uint16_t *h_in = new uint16_t[W*H];
    uint16_t *h_out = new uint16_t[W*H];
    memset(h_out, 0, W*H*sizeof(uint16_t));
    for (int i = 0; i < W*H; i++) h_in[i] = (uint16_t)((i%7)+1)*10;

    launchKernel2D(S_sten3_u16_2d, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   (uint16_t*)h_in, h_out, W, H);

    bool ok = true;
    for (int row = 1; row < H - 1; row++) {
        for (int col = 0; col < W; col++) {
            int c = row * W + col;
            if (abs((int)h_out[c]-(int)(uint16_t)(h_in[c-W] + h_in[c] + h_in[c+W])) > 0) {
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
