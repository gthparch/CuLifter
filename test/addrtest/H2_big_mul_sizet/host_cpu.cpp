#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_H2_big_mul_sizet();

int main() {
    constexpr int N = 1024;
    // Launch with 1 block, 256 threads (so idx = 0*BIG + tid = tid)
    // idx % N maps back to [0..255]
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    float *in = new float[N];
    float *out = new float[N];
    std::memset(out, 0, N * sizeof(float));
    for (int i = 0; i < N; i++) in[i] = (float)(i % 7 + 1) * 0.1f;

    launchKernel(addr_H2_big_mul_sizet, gridDim, blockDim,
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
