// host_cpu.cpp — drive the lifted fp8 V-transpose kernel on x86warp and
// compare the resulting Vt smem against an analytic CPU transpose.
//
// V[s][h] = s*16+h (row-major [SEQ=16][HD=16] e4m3 bytes). A correct
// transpose must give Vt[h][s] = V[s][h] = s*16+h. If the lifted
// LDSM.16.MT88 + PRMT + STSM.16.M88 sequence mishandles the fp8 (8-bit)
// transpose, half the hdim rows of Vt will be wrong/zero — the FA3 e4m3
// V-transpose signature (hdim blocks 1,3 wrong while 0,2 are correct).

#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void kern_fp8_vt();
extern "C" int32_t shared_mem[];   // wrapper's per-CTA shared segment

#define SEQ 16
#define HD  32

int main() {
    uint8_t* hV  = new uint8_t[SEQ * HD];
    uint8_t* hVt = new uint8_t[HD * SEQ];
    std::memset(hVt, 0, HD * SEQ);
    // PROBE modes (structural decode of the transpose permutation, unambiguous
    // unlike the wrapping s*HD+h): PROBE=H -> V[s][h]=h (output reveals source h'),
    // PROBE=S -> V[s][h]=s (reveals source s'). Default = s*HD+h (legacy check).
    const char* probe = getenv("PROBE");
    for (int s = 0; s < SEQ; s++)
        for (int h = 0; h < HD; h++) {
            uint8_t v;
            if (probe && probe[0]=='H')      v = (uint8_t)h;
            else if (probe && probe[0]=='S') v = (uint8_t)s;
            else                             v = (uint8_t)(s * HD + h);
            hV[s * HD + h] = v;
        }
    if (probe && (probe[0]=='H' || probe[0]=='S')) {
        launchKernel(kern_fp8_vt, 1, 32, (const uint8_t*)hV, hVt);
        std::printf("=== PROBE=%c : Vt[h][s] (raw source %s') ===\n", probe[0],
                    probe[0]=='H' ? "h" : "s");
        for (int h = 0; h < HD; h++) {
            std::printf("h=%2d: ", h);
            for (int s = 0; s < SEQ; s++) std::printf("%3d ", hVt[h*SEQ+s]);
            std::printf("\n");
        }
        delete[] hV; delete[] hVt; return 0;
    }
    if (probe && probe[0]=='M') {   // dump smem vsrc region (0..511) after kernel
        for (int s = 0; s < SEQ; s++)
            for (int h = 0; h < HD; h++)
                hV[s*HD+h] = (uint8_t)(s*HD+h);
        launchKernel(kern_fp8_vt, 1, 32, (const uint8_t*)hV, hVt);
        const uint8_t* sm = (const uint8_t*)shared_mem;
        std::printf("=== scan smem for nonzero 16-byte runs (offset : 16 bytes) ===\n");
        for (int off = 0; off < 4096; off += 16) {
            int nz = 0; for (int j=0;j<16;j++) if (sm[off+j]) nz++;
            if (nz) {
                std::printf("off=%4d: ", off);
                for (int j=0;j<16;j++) std::printf("%3d ", sm[off+j]);
                std::printf("\n");
            }
        }
        delete[] hV; delete[] hVt; return 0;
    }

    launchKernel(kern_fp8_vt, /*grid*/ 1, /*block*/ 32,
                 (const uint8_t*)hV, hVt);

    int nmatch = 0, ntot = 0, nzero = 0;
    int firstbad = -1;
    // Per-hdim-block coverage: blocks of 8 hdim (matches the 8x8 fragment).
    int blkmatch[2] = {0, 0};  // even/odd 8-hdim halves within each 8x8 matrix col
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

    std::cout << "=== lifted Vt[h][s] (expect s*16+h) ===\n";
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
