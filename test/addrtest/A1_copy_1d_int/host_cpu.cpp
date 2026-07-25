#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void addr_A1_copy_1d_int();

int main() {
    constexpr int N = 1024;
    constexpr int blockDim = 256;
    constexpr int gridDim = (N + blockDim - 1) / blockDim;

    int *in = new int[N];
    int *out = new int[N];
    std::memset(out, 0, N * sizeof(int));
    for (int i = 0; i < N; i++) in[i] = (int)(i % 7 + 1);

    launchKernel(addr_A1_copy_1d_int, gridDim, blockDim,
                 (const int*)in, out, N);

    bool ok = true;
    for (int i = 0; i < N; i++) {
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
