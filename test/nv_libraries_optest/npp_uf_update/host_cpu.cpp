// host_cpu.cpp — x86 exec-correctness for a REAL npp connected-components kernel.
//
// Kernel: labelMarkersUFUpdate(unsigned int* labels, unsigned int n, NppiSize)
//   _Z20labelMarkersUFUpdatePjj8NppiSize  (from libnpp sm_90 npp_uf.cubin)
//
// Decoded (full SASS): 1D, gid = blockDim.x*blockIdx.x + threadIdx.x.
//   if (gid >= n) return;                       // n = param @0x218
//   x = labels[gid]; if (x >= n) return;
//   for (cnt=0; ; ) { y = labels[x]; if (x==y) break;   // root reached
//                     ++cnt; x = y; if (x>=n) return; if (cnt>=n) return; }
//   labels[gid] = x;                            // path-flatten: point gid at its root
// Union-find "flatten": every node ends pointing at its root. The final state is
// ORDER-INDEPENDENT (compressing labels[g] to root(g) never changes any root), so
// the wrapper's sequential-CTA execution matches a reference computed from the
// ORIGINAL forest. Params: labels@0x210, n@0x218 (NppiSize unused by the kernel).
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void _Z20labelMarkersUFUpdatePjj8NppiSize();

int main() {
    constexpr unsigned N = 200;
    uint32_t lab[N], ref[N];

    // Build a discriminating forest: a root every 10 nodes (0,10,20,…); every
    // other node points one step down toward its block's root. Distinct roots
    // (floor(i/10)*10) and chains of depth up to 9 — a "write 0" kernel fails.
    uint32_t par[N];
    for (unsigned i = 0; i < N; ++i)
        par[i] = (i % 10 == 0) ? i : (i - 1);     // root every 10th, else chain down
    for (unsigned i = 0; i < N; ++i) lab[i] = par[i];

    // Reference: root via the ORIGINAL forest (cap N hops, matches kernel's failsafe).
    for (unsigned i = 0; i < N; ++i) {
        uint32_t x = par[i];
        bool wrote = true;
        if (x >= N) { ref[i] = lab[i]; continue; }          // (won't happen here)
        unsigned cnt = 0;
        while (true) { uint32_t y = par[x]; if (x == y) break; ++cnt; x = y;
                       if (x >= N || cnt >= N) { wrote = false; break; } }
        ref[i] = wrote ? x : lab[i];
    }

    launchKernel(_Z20labelMarkersUFUpdatePjj8NppiSize,
                 (N + 63) / 64, 64,                 // 1D, bounds-guarded (gid>=n exits)
                 ArgAt{0x210}, (uint32_t*)lab,
                 ArgAt{0x218}, (uint32_t)N);

    int bad = 0;
    for (unsigned i = 0; i < N; ++i)
        if (lab[i] != ref[i]) {
            if (bad < 8) printf("  mismatch[%u]: got %u want %u\n", i, lab[i], ref[i]);
            ++bad;
        }
    printf("=== labelMarkersUFUpdate: root of nodes 5,15,25,...,195 ===\n  ");
    for (int i = 5; i < (int)N; i += 10) printf("%u ", lab[i]); printf("\n");
    printf("mismatches = %d / %u\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
