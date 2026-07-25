#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_stsm_x1();

int main() {
    constexpr int N = 256;
    constexpr int blockDim = 256;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[N];
    uint32_t *out_arr = new uint32_t[N];
    std::memset(out_arr, 0, N * sizeof(uint32_t));
    for (int i = 0; i < N; i++) in_arr[i] = 0xD0000000u + i;

    launchKernel(kern_stsm_x1, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // For single-threaded x86: thread 0 loads in_arr[0] from global,
    // STSM.16.M88 stores it to smem[0], then LDS reads smem[0] back.
    // out_arr[0] should equal in_arr[0].
    bool ok = (out_arr[0] == in_arr[0]);
    if (!ok) {
        std::cout << "FAIL: out[0] = " << std::hex << out_arr[0]
                  << ", expected " << in_arr[0] << std::endl;
    }
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
