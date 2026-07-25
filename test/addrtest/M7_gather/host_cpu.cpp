#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void M7_gather();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *data = new float[N];
    int *indices = new int[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        data[i] = (float)(i % 7 + 1) * 0.1f;
        indices[i] = (i * 7) % N;
    }

    launchKernel(M7_gather, gridDim, blockDim,
                 (const float*)data, (const int*)indices, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (out[i] != data[indices[i]]) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << data[indices[i]] << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] data; delete[] indices; delete[] out;
    return ok ? 0 : 1;
}
