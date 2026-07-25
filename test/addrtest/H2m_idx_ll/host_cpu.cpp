#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void H2m_idx_ll();

int main() {
    constexpr int N = 1024;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)((i % 7 + 1) * 0.1f);

    Dim3 grid{1, 1};
    Dim3 block{256, 1};
    launchKernel2D(H2m_idx_ll, grid, block,
                   (const float*)in, out, N);

    bool ok = true;
    for (int i = 0; i < 256; i++) {
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
