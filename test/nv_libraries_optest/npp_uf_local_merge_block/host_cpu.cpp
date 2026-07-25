// host_cpu.cpp — x86 exec-correctness for a REAL npp connected-components kernel.
//
// Kernel: labelMarkersUF8WayLocalMergeBlock<unsigned char>(
//             unsigned char* image, size_t pitch, unsigned int* labels, NppiSize)
//   _Z33labelMarkersUF8WayLocalMergeBlockIhEvPT_mPj8NppiSize  (libnpp sm_90 npp_uf.cubin)
//
// Decoded (full SASS): 2D, block 16x16, grid ceil(w/16)xceil(h/16). This is the
// INTRA-TILE init step. Each block handles ONE 16x16 tile independently:
//   col = ctaid.x*16+tid.x; row = ctaid.y*16+tid.y;
//   if (col>=width@0x228 || row>=height@0x22c) return;
//   - load image[row*pitch+col] into shared mem (per-tile),
//   - run an 8-connected union-find ENTIRELY in shared memory (STS/LDS + ATOMS.MIN)
//     synchronised by BAR.SYNC, merging only pixels WITHIN this tile that share the
//     same pixel value (local indices, neighbours -0x11/-0x10/-0x1 = up-left/up/left
//     within the 16-wide tile),
//   - the local root (a 16x16-local index) is remapped to a GLOBAL linear index:
//       rootColGlobal = (ctaid.x*16) | (localRoot & 0xf)
//       rootRowGlobal = (ctaid.y*16) +  (localRoot >> 4)
//       labels[row*width+col] = rootRowGlobal*width + rootColGlobal
//
// CRITICAL — the local step uses ONLY 3 backward neighbours {up-left, up, left}
// (offsets -0x11, -0x10, -0x1 in the SASS), NOT the up-right neighbour. So a
// down-left diagonal staircase (corner-linked via an UP-RIGHT touch) is left
// SPLIT by this kernel on purpose — those edges are resolved by the later boundary/
// global passes. The reference must use exactly {up-left, up, left}.
//
// ORDER-INDEPENDENCE: tiles are disjoint — block B reads only the read-only image
// and writes only its own tile's labels[]; no cross-tile sharing. So the OUTPUT is
// order-independent ACROSS CTAs. WITHIN a tile the algorithm is barrier-synchronised
// (BAR.SYNC) and applies the min-index hooking merge over a FIXED edge set, so it is
// deterministic (verified: stable across repeated runs). Reference from the ORIGINAL
// image: per tile, union-find over {up-left,up,left} equal-value edges, root = MIN
// tile-local index (= MIN global linear index within the tile), remapped to global.
//
// Params: image@0x210, pitch@0x218, labels@0x220, width@0x228, height@0x22c.
#include <cstdint>
#include <cstdio>
#include <vector>
#include <functional>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void _Z33labelMarkersUF8WayLocalMergeBlockIhEvPT_mPj8NppiSize();

static const int TS = 16;  // tile size

int main() {
    // Image larger than one tile so we exercise MULTIPLE independent tiles.
    constexpr int W = 36, H = 20;     // 3x2 tiles (last tiles partial)
    constexpr int N = W * H;
    std::vector<uint8_t> img(N);
    // Structured pattern with components that stay within tiles and some that would
    // cross tile borders (those must be SPLIT at the tile boundary by this step).
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        // checker-ish blobs of size ~3, value depends on coarse cell
        img[y*W + x] = (uint8_t)(((x/3) + (y/3)) % 4);
      }

    std::vector<uint32_t> lab(N, 0xdeadbeef);

    // ---- Reference: per-tile union-find over {up-left, up, left} edges ----
    // parent[] is a global-linear-index forest; we only ever union pixels in the same
    // tile, and only via the 3 backward neighbours the kernel checks. Root = min
    // global linear index of the (tile-restricted) component = the value the kernel
    // remaps and stores (since min local-16 index == min global linear index in-tile).
    std::vector<int> parent(N);
    for (int i = 0; i < N; ++i) parent[i] = i;
    std::function<int(int)> find = [&](int a) {
        while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; }
        return a;
    };
    auto uni = [&](int a, int b) {
        int ra = find(a), rb = find(b);
        if (ra < rb) parent[rb] = ra; else if (rb < ra) parent[ra] = rb;
    };
    const int bdx[3] = {-1, 0, -1};   // up-left, up, left
    const int bdy[3] = {-1, -1, 0};
    for (int ty0 = 0; ty0 < H; ty0 += TS) {
      for (int tx0 = 0; tx0 < W; tx0 += TS) {
        int txEnd = std::min(tx0 + TS, W), tyEnd = std::min(ty0 + TS, H);
        for (int y = ty0; y < tyEnd; ++y)
          for (int x = tx0; x < txEnd; ++x) {
            int p = y*W + x; uint8_t v = img[p];
            for (int k = 0; k < 3; ++k) {
              int nx = x + bdx[k], ny = y + bdy[k];
              if (nx < tx0 || nx >= txEnd || ny < ty0 || ny >= tyEnd) continue; // stay in tile
              if (img[ny*W + nx] == v) uni(p, ny*W + nx);
            }
          }
      }
    }
    std::vector<uint32_t> ref(N);
    for (int p = 0; p < N; ++p) ref[p] = (uint32_t)find(p);   // store global-linear root

    const int BX = 16, BY = 16;
    launchKernel2D(_Z33labelMarkersUF8WayLocalMergeBlockIhEvPT_mPj8NppiSize,
                   Dim3{(W+BX-1)/BX, (H+BY-1)/BY}, Dim3{BX, BY},
                   ArgAt{0x210}, (uint8_t*)img.data(),
                   ArgAt{0x218}, (uint64_t)W,
                   ArgAt{0x220}, (uint32_t*)lab.data(),
                   ArgAt{0x228}, (int32_t)W,
                   ArgAt{0x22c}, (int32_t)H);

    int bad = 0;
    for (int p = 0; p < N; ++p)
        if (lab[p] != ref[p]) {
            if (bad < 12)
                printf("  mismatch p=%d (r%d,c%d val=%d): got %u want %u\n",
                       p, p/W, p%W, img[p], lab[p], ref[p]);
            ++bad;
        }
    printf("=== local_merge_block: %dx%d image, per-tile {up-left,up,left} UF roots ===\n", W, H);
    printf("mismatches = %d / %d\n", bad, N);
    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
