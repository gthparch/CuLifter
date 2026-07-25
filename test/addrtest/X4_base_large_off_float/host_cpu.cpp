#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_X4_base_large_off_float();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    float *in1 = new float[N];
    float *in2 = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        in1[i] = (float)(i % 7 + 1) * 0.1f;
        in2[i] = (float)((i % 5 + 1) * 0.3f);
    }

    int selector = 0;
    unsigned long long base_offset = 0ULL;
    launchKernel(addr_X4_base_large_off_float, gridDim, blockDim,
                 (const float*)in1, (const float*)in2, out, N, selector, base_offset);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        float expected = in1[i];  // selector == 0
        if (out[i] != expected) {
            std::cout << "FAIL at " << i << ": got " << out[i]
                      << " expected " << expected << "\n";
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "TEST PASSED" : "FAIL") << std::endl;

    delete[] in1; delete[] in2; delete[] out;
    return ok ? 0 : 1;
}
