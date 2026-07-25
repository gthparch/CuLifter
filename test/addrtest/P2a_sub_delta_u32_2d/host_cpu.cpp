#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2a_sub_delta_u32_2d();

int main() {
    constexpr int W = 32, H = 32, row_delta = 2;
    uint32_t *h_in = new uint32_t[W*H];
    uint32_t *h_out = new uint32_t[W*H];
    memset(h_out, 0, W*H*sizeof(uint32_t));
    for (int i = 0; i < W*H; i++) h_in[i] = (uint32_t)((i%7+1)*100);

    launchKernel2D(P2a_sub_delta_u32_2d, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   (uint32_t*)h_in, h_out, W, H, row_delta);

    bool ok = true;
    for (int row = row_delta; row < H; row++)
        for (int col = 0; col < W; col++) {
            int c = row*W+col;
            if (h_out[c] != h_in[(row-row_delta)*W+col]) {
                std::cout << "FAIL at row=" << row << " col=" << col << std::endl;
                ok = false; goto done;
            }
        }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
