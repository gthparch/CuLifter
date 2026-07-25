#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_stsm_x2();

int main() {
    constexpr int N = 128;
    constexpr int IN_ELEMS = N * 2;
    constexpr int OUT_ELEMS = N * 4;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[IN_ELEMS];
    uint32_t *out_arr = new uint32_t[OUT_ELEMS];
    std::memset(out_arr, 0, OUT_ELEMS * sizeof(uint32_t));
    for (int i = 0; i < IN_ELEMS; i++) in_arr[i] = 0xD0000000u + i;

    launchKernel(kern_stsm_x2, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // x2 stores only matrices 0-1 through lanes 0-15's addresses; the
    // addresses of lanes 16-31 are IGNORED (PTX ISA) — words 64..127 of
    // each warp window are a canary that must stay zero. A per-lane
    // contiguous lowering stores through all 32 lanes and clobbers it.
    int n_fail = 0, first = -1;
    auto check = [&](int idx, uint32_t expect) {
        if (out_arr[idx] != expect) {
            if (first < 0) first = idx;
            n_fail++;
        }
    };
    for (int wq = 0; wq < blockDim / 32; wq++) {
        for (int m = 0; m < 2; m++)
            for (int r = 0; r < 8; r++)
                for (int w = 0; w < 4; w++)
                    check((32 * wq + 8 * m + r) * 4 + w,
                          in_arr[(32 * wq + 4 * r + w) * 2 + m]);
        for (int k = 64; k < 128; k++)
            check(128 * wq + k, 0u);
    }

    bool ok = (n_fail == 0);
    if (!ok)
        std::cout << "FAIL: " << n_fail << "/" << OUT_ELEMS
                  << " words differ; first at " << first
                  << " got=0x" << std::hex << out_arr[first] << std::dec
                  << std::endl;
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
