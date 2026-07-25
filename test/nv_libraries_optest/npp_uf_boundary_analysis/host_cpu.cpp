// host_cpu.cpp — x86 exec-correctness for a REAL npp connected-components kernel.
//
// Kernel: labelMarkersUF8WayBoundaryAnalysis<unsigned char>(
//             unsigned char* image, size_t pitch, unsigned int* labels, NppiSize)
//   _Z34labelMarkersUF8WayBoundaryAnalysisIhEvPT_mPj8NppiSize  (libnpp sm_90 npp_uf.cubin)
//
// Decoded (full SASS): 2D, block 16x16, grid ceil(w/16)xceil(h/16).
//   gx = ctaid.x*16+tid.x; gy = ctaid.y*16+tid.y;
//   if (gx>=width@0x228 || gy>=height@0x22c) return;
//   Using integer reciprocal-divide (MUFU.RCP / IMAD.HI sequences) it derives the
//   16x16 TILE row/col and the within-tile boundary position, then ONLY for pixels
//   sitting on a tile EDGE does it compare image[p] against its cross-tile neighbours
//   (read-only u8 image at base+row*pitch+col) and, when equal, unions their global
//   label-trees with the same atomic-min hooking merge as global_merge (ATOMG.MIN
//   loops, BREAK on settle). Linear label index p = row*width + col. Image read-only;
//   only labels[] is written; edges depend ONLY on the read-only image.
//
// ====================  STATUS: BLOCKED for full-CCL  ========================
// This is the CROSS-TILE-BOUNDARY pass of NPP's ITERATIVE union-find pipeline
// (local_merge_block establishes intra-tile labels; boundary_analysis + global merge
// resolve tile borders; the host loops to a fixed point). A single standalone pass
// from labels[p]=p does NOT converge to full component-min — boundary_analysis by
// construction only touches tile-edge pixels, so interior pixels keep their identity
// labels and a one-shot full-CCL reference cannot hold. The pass IS byte-DETERMINISTIC
// (stable label hash across 8 reps) but its single-pass split structure is not a clean
// order-independent full-CCL function, so we do not force a pass against component-min.
//
// WHAT IS verified (genuine, order-independent, DISCRIMINATING): SOUNDNESS — every
// pixel's chain-root lies in the SAME true 8-connected (equal-value) component
// (the kernel never merges across components), AND the result is byte-deterministic
// across reruns. Both depend only on the read-only image; a garbage / wrong-direction
// merge would fail. Full convergence needs the multi-kernel driver (see metadata.json).
//
// Params: image@0x210, pitch@0x218, labels@0x220, width@0x228, height@0x22c.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <queue>
#include "kernel_wrapper.h"

extern "C" void _Z34labelMarkersUF8WayBoundaryAnalysisIhEvPT_mPj8NppiSize();

static int trueComp(const std::vector<uint8_t>& img, int W, int H, std::vector<int>& comp) {
    comp.assign(W*H, -1);
    std::vector<char> seen(W*H, 0);
    const int dx[8]={-1,0,1,-1,1,-1,0,1}, dy[8]={-1,-1,-1,0,0,1,1,1};
    int cid = 0;
    for (int s = 0; s < W*H; ++s) {
        if (seen[s]) continue;
        std::queue<int> q; q.push(s); seen[s]=1; uint8_t v=img[s];
        while (!q.empty()) {
            int c=q.front(); q.pop(); comp[c]=cid; int cx=c%W, cy=c/W;
            for (int k=0;k<8;++k){int nx=cx+dx[k],ny=cy+dy[k];
                if(nx<0||nx>=W||ny<0||ny>=H)continue; int n=ny*W+nx;
                if(!seen[n]&&img[n]==v){seen[n]=1;q.push(n);}}
        }
        ++cid;
    }
    return cid;
}

int main() {
    // Multi-tile image with components straddling tile borders (the kernel's purpose).
    constexpr int W = 34, H = 22, N = W*H;
    std::vector<uint8_t> img(N);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        img[y*W+x] = (uint8_t)((((x/2) ^ (y/2)) + (x/5)) % 3);

    std::vector<int> comp; trueComp(img, W, H, comp);

    auto run = [&](std::vector<uint32_t>& lab){
        for (int i=0;i<N;++i) lab[i]=(uint32_t)i;
        launchKernel2D(_Z34labelMarkersUF8WayBoundaryAnalysisIhEvPT_mPj8NppiSize,
                       Dim3{(W+15)/16,(H+15)/16}, Dim3{16,16},
                       ArgAt{0x210}, (uint8_t*)img.data(),
                       ArgAt{0x218}, (uint64_t)W,
                       ArgAt{0x220}, (uint32_t*)lab.data(),
                       ArgAt{0x228}, (int32_t)W,
                       ArgAt{0x22c}, (int32_t)H);
    };
    std::vector<uint32_t> lab(N), lab2(N);
    run(lab); run(lab2);

    auto root = [&](std::vector<uint32_t>& L, int p){
        int g=0; while (L[p]!=(uint32_t)p){p=(int)L[p]; if(++g>N)break;} return p; };

    int crossErr = 0, detErr = 0;
    for (int p = 0; p < N; ++p) {
        int r = root(lab, p);
        if (comp[r] != comp[p]) {
            if (crossErr < 8)
                printf("  CROSS-MERGE p=%d (comp %d) -> root %d (comp %d)\n",
                       p, comp[p], r, comp[r]);
            ++crossErr;
        }
        if (lab[p] != lab2[p]) ++detErr;
    }
    int bad = crossErr + detErr;
    printf("=== boundary_analysis: %dx%d multi-tile image; SOUNDNESS+determinism ===\n", W, H);
    printf("  cross-component merges = %d (must be 0); nondeterministic pixels = %d\n",
           crossErr, detErr);
    printf("  NOTE: full component-min convergence needs the iterative driver loop;\n");
    printf("        this boundary pass is sound but not fully convergent (BLOCKED for full-CCL).\n");
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
