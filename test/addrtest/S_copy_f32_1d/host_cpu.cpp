#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_copy_f32_1d();

int main() {
    constexpr int N = 1024;
    float *h_in = new float[N];
    float *h_out = new float[N];
    memset(h_out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)((i%7)+1)*0.1f;

    launchKernel(S_copy_f32_1d, (N + 255) / 256, 256, (float*)h_in, h_out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
