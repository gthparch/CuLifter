#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_ldsm_x4();

int main() {
    constexpr int N = 128;
    constexpr int ELEMS = N * 4;
    constexpr int blockDim = 128;
    constexpr int gridDim = 1;

    uint32_t *in_arr = new uint32_t[ELEMS];
    uint32_t *out_arr = new uint32_t[ELEMS];
    std::memset(out_arr, 0, ELEMS * sizeof(uint32_t));
    for (int i = 0; i < ELEMS; i++) in_arr[i] = 0xC0000000u + i;

    launchKernel(kern_ldsm_x4, gridDim, blockDim,
                 (const uint32_t*)in_arr, out_arr, N);

    // Exact compare against the GPU capture (ref_output.bin): ldmatrix.x4
    // redistributes fragments ACROSS lanes — reg m of lane L is the 4-byte
    // slice (L&3) of the 16-byte row addressed by lane 8m+(L>>2). With the
    // distinct 0xC0000000+i fill, the capture pins the exact distribution;
    // a per-lane contiguous load matches it in only 8/512 elements.
    std::ifstream rf("ref_output.bin", std::ios::binary);
    bool ok = true;
    if (!rf) {
        std::cout << "FAIL: cannot open ref_output.bin" << std::endl;
        ok = false;
    } else {
        uint32_t ref[ELEMS];
        rf.read((char*)ref, sizeof(ref));
        if (rf.gcount() != (std::streamsize)sizeof(ref)) {
            std::cout << "FAIL: ref_output.bin short read" << std::endl;
            ok = false;
        } else {
            int n_fail = 0, first = -1;
            for (int i = 0; i < ELEMS; i++) {
                if (out_arr[i] != ref[i]) {
                    if (first < 0) first = i;
                    n_fail++;
                }
            }
            if (n_fail) {
                std::cout << "FAIL: " << n_fail << "/" << ELEMS
                          << " words differ; first at " << first
                          << " got=0x" << std::hex << out_arr[first]
                          << " ref=0x" << ref[first] << std::dec << std::endl;
                ok = false;
            }
        }
    }
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;

    delete[] in_arr; delete[] out_arr;
    return ok ? 0 : 1;
}
