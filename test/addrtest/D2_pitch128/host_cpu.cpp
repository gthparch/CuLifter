#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void D2_pitch128();

int main() {
    constexpr int W = 32, H = 32;

    constexpr int pitch = 128;
    float *h_in = new float[pitch*H];
    float *h_out = new float[pitch*H];
    memset(h_out, 0, pitch*H*sizeof(float));
    for (int i = 0; i < pitch*H; i++) h_in[i] = (float)(i % 7 + 1) * 0.1f;
    launchKernel2D(D2_pitch128, Dim3{(32+15)/16, (32+15)/16}, Dim3{16, 16},
                   h_in, h_out, pitch, W, H);

    bool ok = true;
    for (int row = 1; row < H-1; row++) for (int col = 0; col < W; col++) {
        int c = row*pitch+col;
        float expected = h_in[c-pitch]+h_in[c]+h_in[c+pitch];
        if (fabsf(h_out[c]-expected) > 1e-5f) {
            std::cout << "FAIL at row=" << row << " col=" << col << std::endl;
            ok = false; goto done;
        }
    }
    done:
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
