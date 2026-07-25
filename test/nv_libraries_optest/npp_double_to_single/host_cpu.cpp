// host_cpu.cpp — x86 execution-correctness driver for a REAL NPP library kernel.
//
// Kernel: npp::kernelDoubleToSingleList(short2* A, short2* B, short2* C, int n)
//   (extracted from libnpp sm_90 cubin, lifted to x86 by SLifter).
//
// Decoded semantics (from SASS + lifted IR):
//   gid = blockDim.x*blockIdx.x + threadIdx.x        (1D launch; blockIdx.y == 0)
//   C[gid].x = A[gid].x        (low  u16 of A[gid], loaded at A + gid*4)
//   C[gid].y = B[gid].y        (high u16 of B[gid], loaded at B + gid*4 + 2)
//   packed by PRMT 0x5410 into one 32-bit store at C + gid*4.
// No bounds check in the kernel, so we launch exactly N threads for N elements.
//
// Params live in the SM90 const bank: A@0x210, B@0x218, C@0x220, n@0x228.
// The kernel's desc[UR][addr] global access uses base 0 (lifter drops the
// default global descriptor), so the raw pointers go straight in.
//
// Reference is computed independently on the host (trivial interleave), so this
// is a genuine semantic check of the lifted library kernel — no GPU needed.
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z24kernelDoubleToSingleListP6short2S0_S0_i();

struct short2 { uint16_t x, y; };

int main() {
    constexpr int BLOCK = 64, GRID = 4;
    constexpr int N = BLOCK * GRID;   // 256 elements, one per thread

    short2 A[N], B[N], C[N], ref[N];
    for (int i = 0; i < N; ++i) {
        // distinct values in every lane so a wrong byte-pack is caught
        A[i].x = (uint16_t)(i * 7 + 1);
        A[i].y = (uint16_t)(i * 7 + 2);
        B[i].x = (uint16_t)(i * 13 + 3);
        B[i].y = (uint16_t)(i * 13 + 4);
        C[i].x = 0xDEAD; C[i].y = 0xBEEF;       // poison
        ref[i].x = A[i].x;                       // expected: x from A
        ref[i].y = B[i].y;                       // expected: y from B
    }

    launchKernel(_Z24kernelDoubleToSingleListP6short2S0_S0_i, GRID, BLOCK,
                 ArgAt{0x210}, (short2*)A,
                 ArgAt{0x218}, (short2*)B,
                 ArgAt{0x220}, (short2*)C,
                 ArgAt{0x228}, (int)N);

    int bad = 0;
    for (int i = 0; i < N; ++i)
        if (C[i].x != ref[i].x || C[i].y != ref[i].y) {
            if (bad < 8)
                printf("  mismatch[%d]: got (%u,%u) want (%u,%u)\n",
                       i, C[i].x, C[i].y, ref[i].x, ref[i].y);
            ++bad;
        }

    printf("=== npp kernelDoubleToSingleList: out[0..3] ===\n");
    for (int i = 0; i < 4; ++i) printf("  C[%d]=(%u,%u)\n", i, C[i].x, C[i].y);
    printf("mismatches = %d / %d\n", bad, N);

    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
