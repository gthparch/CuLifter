#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void D2_base_selected();

int main() {
    constexpr int W = 32, H = 32;

    float *h_in1 = new float[W*H];
    float *h_in2 = new float[W*H];
    float *h_out = new float[W*H];
    memset(h_out, 0, W*H*sizeof(float));
    for (int i = 0; i < W*H; i++) { h_in1[i] = (float)(i%7+1)*0.1f; h_in2[i] = (float)(i%5+1)*0.2f; }
    launchKernel2D(D2_base_selected, Dim3{(W+15)/16, (H+15)/16}, Dim3{16, 16},
                   h_in1, h_in2, h_out, W, H, 0);

    bool ok = true;
    for (int row = 1; row < H-1; row++) for (int col = 0; col < W; col++) {
        int c = row*W+col;
        float expected = h_in1[c-W]+h_in1[c]+h_in1[c+W];  // selector=0 → in1
        if (fabsf(h_out[c]-expected) > 1e-5f) {
            std::cout << "FAIL at row=" << row << " col=" << col << std::endl;
            ok = false; goto done;
        }
    }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in1; delete[] h_in2; delete[] h_out;
    return ok ? 0 : 1;
}
