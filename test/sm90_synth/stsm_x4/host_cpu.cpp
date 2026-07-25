#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_stsm_x4();

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 4;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[ELEMS];
    uint32_t *out_arr = new uint32_t[2 * ELEMS];
    std::memset(out_arr, 0, 2 * ELEMS * sizeof(uint32_t));
    for (int i = 0; i < ELEMS; i++) in_arr[i] = 0xC0000000u + i;

    launchKernel(kern_stsm_x4, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // st.matrix layout (PTX ISA): word w of the 16B row addressed by lane
    // 8m+r = source reg m of lane 4r+w. The kernel's main() asserts the
    // same formula on real hardware, so this comparator is HW-checkable.
    // Second half: ldmatrix.x4 round-trip of the stored rows == identity.
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
                for (int w = 0; w < 4; w++)
                    check((32 * wq + 8 * m + r) * 4 + w,
                          in_arr[(32 * wq + 4 * r + w) * 4 + m]);
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
