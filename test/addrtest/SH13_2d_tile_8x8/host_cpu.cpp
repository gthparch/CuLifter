#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH13_2d_tile_8x8();

int main() {
    constexpr int W = 32, H = 32;
    float *h_in = new float[W*H], *h_out = new float[W*H];
    memset(h_out, 0, W*H*sizeof(float));
    for (int i = 0; i < W*H; i++) h_in[i] = (float)(i*3+7);
    launchKernel2D(SH13_2d_tile_8x8, Dim3{(W+7)/8, (H+7)/8}, Dim3{8, 8}, h_in, h_out, W, H);
    bool ok = true;
    for (int i = 0; i < W*H; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
