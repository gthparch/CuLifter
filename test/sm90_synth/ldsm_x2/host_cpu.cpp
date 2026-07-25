#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_ldsm_x2();

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 2;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[ELEMS];
    uint32_t *out_arr = new uint32_t[ELEMS];
    std::memset(out_arr, 0, ELEMS * sizeof(uint32_t));
    for (int i = 0; i < ELEMS; i++) in_arr[i] = 0xB0000000u + i;

    launchKernel(kern_ldsm_x2, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // Exact compare against the ldmatrix layout (fragment m of lane L =
    // word (L&3) of the row addressed by lane 8m+(L>>2); rows here are at
    // smem[(16w + (L&15))*4]):
    //   out[(32w+L)*2+m] = in[(16w + 8m + (L>>2))*4 + (L&3)].
    // The formula is HW-pinned by the ldsm_x4 GPU capture (512/512 match)
    // and re-asserted on hardware by this kernel's GPU main().
    int n_fail = 0, first = -1;
    for (int w = 0; w < blockDim / 32; w++)
        for (int L = 0; L < 32; L++)
            for (int m = 0; m < 2; m++) {
                uint32_t expect = in_arr[(16 * w + 8 * m + (L >> 2)) * 4 + (L & 3)];
                int idx = (32 * w + L) * 2 + m;
                if (out_arr[idx] != expect) {
                    if (first < 0) first = idx;
                    n_fail++;
                }
            }

    bool ok = (n_fail == 0);
    if (!ok)
        std::cout << "FAIL: " << n_fail << "/" << ELEMS
                  << " words differ; first at " << first
                  << " got=0x" << std::hex << out_arr[first] << std::dec
                  << std::endl;
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
