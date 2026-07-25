#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_stsm_mt88();

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 4;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[ELEMS];
    uint32_t *out_arr = new uint32_t[2 * ELEMS];
    std::memset(out_arr, 0, 2 * ELEMS * sizeof(uint32_t));
    for (int i = 0; i < ELEMS; i++)
        in_arr[i] = (0xA000u + 2 * i) | ((0xB001u + 2 * i) << 16);

    launchKernel(kern_stsm_mt88, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // Transposed st.matrix layout: memory receives F^T —
    //   word w of the row addressed by lane 8m+r =
    //     sel(r&1, frag_m[8w+(r>>1)]) | sel(r&1, frag_m[8w+4+(r>>1)]) << 16
    // (sel(0)=lo16, sel(1)=hi16). Verified on HW by the kernel's main().
    // Second half: ldmatrix.x4.trans round-trip == identity.
    auto frag = [&](int wq, int lane, int m) {
        return in_arr[(32 * wq + lane) * 4 + m];
    };
    int n_fail = 0, first = -1;
    auto check = [&](int idx, uint32_t expect) {
        if (out_arr[idx] != expect) {
            if (first < 0) first = idx;
            n_fail++;
        }
    };
    for (int wq = 0; wq < blockDim / 32; wq++)
        for (int m = 0; m < 4; m++)
            for (int r = 0; r < 8; r++)
                for (int w = 0; w < 4; w++) {
                    uint32_t v0 = frag(wq, 8 * w + (r >> 1), m);
                    uint32_t v1 = frag(wq, 8 * w + 4 + (r >> 1), m);
                    uint16_t lo = (uint16_t)((r & 1) ? (v0 >> 16) : (v0 & 0xffff));
                    uint16_t hi = (uint16_t)((r & 1) ? (v1 >> 16) : (v1 & 0xffff));
                    check((32 * wq + 8 * m + r) * 4 + w,
                          (uint32_t)lo | ((uint32_t)hi << 16));
                }
    for (int i = 0; i < ELEMS; i++)
        check(ELEMS + i, in_arr[i]);

    bool ok = (n_fail == 0);
    if (!ok)
        std::cout << "FAIL: " << n_fail << "/" << 2 * ELEMS
                  << " words differ; first at " << first
                  << " got=0x" << std::hex << out_arr[first] << std::dec
                  << std::endl;
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
