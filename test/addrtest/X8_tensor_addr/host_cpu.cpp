#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void X8_tensor_addr();

int main() {
    constexpr int B = 2, Ht = 8, Wt = 8, Ch = 4;
    constexpr int total = B * Ht * Wt * Ch;

    float *in = new float[total];
    float *out = new float[total];
    std::memset(out, 0, total * sizeof(float));
    for (int i = 0; i < total; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    size_t C = Ch;
    size_t H_stride = (size_t)Ht * Wt * Ch;
    size_t W_stride = (size_t)Wt * Ch;

    // 3D grid: grid.x = (Wt+7)/8, grid.y = (Ht+7)/8, grid.z = B
    Dim3 grid{(Wt + 7) / 8, (Ht + 7) / 8, B};
    Dim3 block{8, 8};
    launchKernel2D(X8_tensor_addr, grid, block,
                   (const float*)in, out, C, H_stride, W_stride,
                   B, Ht, Wt, Ch);

    bool ok = true;
    for (int i = 0; i < total; i++) {
        if (std::fabs(out[i] - in[i]) > 1e-5f) {
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
