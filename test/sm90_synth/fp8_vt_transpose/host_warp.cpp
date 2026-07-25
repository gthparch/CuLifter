// host_warp.cpp — race-free driver for the fp8 V-transpose microbench using the
// serialized warp/fiber model (launchKernel_warp). Each lane fiber runs its vsrc
// fill straight-line, then PARKS at the ldsm collective; the collective completes
// only once all 32 lanes have filled their share, so the vsrc store loop can never
// race the ldmatrix load (the per-CUDA-thread launchKernel had no warp barrier for
// the DCE'd __syncwarp). Validates the trans==2 fragment permutation in isolation.
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper_warp.h"

extern "C" void kern_fp8_vt();

#define SEQ 16
#define HD  32

int main() {
    uint8_t* hV  = new uint8_t[SEQ * HD];
    uint8_t* hVt = new uint8_t[HD * SEQ];
    std::memset(hVt, 0, HD * SEQ);
    for (int s = 0; s < SEQ; s++)
        for (int h = 0; h < HD; h++)
            hV[s * HD + h] = (uint8_t)(s * HD + h);

    launchKernel_warp(kern_fp8_vt, /*grid*/ 1, /*block*/ 32,
                      (const uint8_t*)hV, hVt);

    int nmatch = 0, ntot = 0, nzero = 0, firstbad = -1;
    int hdrowmatch[HD];
    for (int h = 0; h < HD; h++) {
        int rm = 0;
        for (int s = 0; s < SEQ; s++) {
            uint8_t got = hVt[h * SEQ + s];
            uint8_t exp = (uint8_t)(s * HD + h);
            ntot++;
            if (got == exp) { nmatch++; rm++; }
            else if (firstbad < 0) firstbad = h * SEQ + s;
            if (got == 0) nzero++;
        }
        hdrowmatch[h] = rm;
    }
    std::cout << "=== lifted Vt[h][s] (expect s*32+h) [WARP/fiber] ===\n";
    for (int h = 0; h < HD; h++) {
        std::printf("h=%2d (match %2d/%d): ", h, hdrowmatch[h], SEQ);
        for (int s = 0; s < SEQ; s++) std::printf("%3d ", hVt[h * SEQ + s]);
        std::printf("\n");
    }
    std::printf("transpose match %d/%d  zeros=%d  firstbad_idx=%d\n",
                nmatch, ntot, nzero, firstbad);
    bool ok = (nmatch == ntot);
    std::cout << (ok ? "TEST PASSED" : "TEST FAILED") << std::endl;
    delete[] hV; delete[] hVt;
    return ok ? 0 : 1;
}
