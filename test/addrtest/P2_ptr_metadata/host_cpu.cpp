#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2_ptr_metadata();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    int stride = 1, offset = 0, scale = 2;

    launchKernel(P2_ptr_metadata, gridDim, blockDim,
                 (const float*)in, out, N, stride, offset, scale);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int src = i * stride + offset;
        if (src >= 0 && src < N) {
            float e = in[src] * (float)scale;
            if (std::fabs(out[i] - e) > 1e-5f) {
                std::cout << "FAIL at " << i << ": got " << out[i]
                          << " expected " << e << "\n";
                ok = false;
                break;
            }
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
