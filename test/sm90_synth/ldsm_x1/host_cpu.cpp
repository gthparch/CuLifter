#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_ldsm_x1();

int main() {
    constexpr int N = 128;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[N];
    uint32_t *out_arr = new uint32_t[N];
    std::memset(out_arr, 0, N * sizeof(uint32_t));
    for (int i = 0; i < N; i++) in_arr[i] = 0xA0000000u + i;

    launchKernel(kern_ldsm_x1, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // Exact compare against the ldmatrix layout (fragment of lane L = word
    // (L&3) of the row addressed by lane (L>>2); rows here are at
    // smem[(8w + (L&7))*4]): out[32w+L] = in[(8w + (L>>2))*4 + (L&3)].
    // The formula is HW-pinned by the ldsm_x4 GPU capture (512/512 match)
    // and re-asserted on hardware by this kernel's GPU main().
    int n_fail = 0, first = -1;
    for (int w = 0; w < blockDim / 32; w++)
        for (int L = 0; L < 32; L++) {
            uint32_t expect = in_arr[(8 * w + (L >> 2)) * 4 + (L & 3)];
            if (out_arr[32 * w + L] != expect) {
                if (first < 0) first = 32 * w + L;
                n_fail++;
            }
        }

    bool ok = (n_fail == 0);
    if (!ok)
        std::cout << "FAIL: " << n_fail << "/" << N
                  << " words differ; first at " << first
                  << " got=0x" << std::hex << out_arr[first] << std::dec
                  << std::endl;
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
