#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void N5_halo_window();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    launchKernel(N5_halo_window, gridDim, blockDim,
                 (const float*)in, out, N);

    bool ok = true;
    for (int i = 4; i < N - 4; i++) {
        float e = 0;
        for (int d = -4; d <= 4; d++) e += in[i + d];
        if (std::fabs(out[i] - e) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << e << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
