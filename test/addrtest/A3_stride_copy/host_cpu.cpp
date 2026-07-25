#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void A3_stride_copy();

int main() {
    constexpr int N = 1024;
    constexpr int stride = 3;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    launchKernel(A3_stride_copy, gridDim, blockDim,
                 (const float*)in, out, N, stride);

    bool ok = true;
    for (int i = 0; i < N && i * stride < N; i++) {
        if (out[i] != in[i * stride]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << in[i * stride] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in; delete[] out;
    return ok ? 0 : 1;
}
