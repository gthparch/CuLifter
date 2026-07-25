// Host driver for the cub_rank microbenchmark (lifted rank_kernel).
// Validity check (rank semantics, independent of intra-digit HW order):
//   (a) ranks across the tile are a permutation of 0..TILE-1
//   (b) sorting keys by rank yields digit-nondecreasing order
//   (c) per-digit rank ranges match the host-computed digit histogram
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void rank_kernel();

#ifndef BT
#define BT 384
#endif
#define IPT 4
#define TILE (BT * IPT)

int main(int argc, char** argv) {
  unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : 12345u;
  uint32_t* in = new uint32_t[TILE];
  int* ranks = new int[TILE];
  int* digits = new int[TILE];
  uint32_t x = seed ? seed : 1u;
  for (int i = 0; i < TILE; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;  // xorshift32
    in[i] = x;
    ranks[i] = -999;
    digits[i] = -1;
  }

  launchKernel(rank_kernel, 1, BT,
               ArgAt{0x210}, in,
               ArgAt{0x218}, ranks,
               ArgAt{0x220}, digits);

  // Post-kernel shared-memory dump (gridDim=1 so final CTA state persists).
  // Lifted shared_mem is BYTE-addressed ([32768 x i8] in the IR); the wrapper
  // symbol is int32_t[], so warp_offsets[w][bin] (SASS byte w*0x400 + bin*4)
  // is int32 element w*0x100 + bin.
  if (getenv("DUMP_SHM")) {
    const int32_t* shm = shared_mem;
    long total_per_warp[12] = {0};
    int nz = 0;
    for (int w = 0; w < 12; ++w)
      for (int bin = 0; bin < 256; ++bin) {
        int v = shm[w * 0x100 + bin];
        if (v) nz++;
        total_per_warp[w] += v;
      }
    printf("warp_offsets region: %d nonzero of 3072\n", nz);
    for (int w = 0; w < 12; ++w)
      printf("  warp %2d sum=%ld  first bins: %d %d %d %d %d %d %d %d\n", w,
             total_per_warp[w], shm[w * 0x100], shm[w * 0x100 + 1],
             shm[w * 0x100 + 2], shm[w * 0x100 + 3], shm[w * 0x100 + 4],
             shm[w * 0x100 + 5], shm[w * 0x100 + 6], shm[w * 0x100 + 7]);
    // expected final warp_offsets[w][bin] (exact, from host histogram):
    // prefix(bin) + hist_warp<=w(bin); compare a few bins of warp 11
    int pfx_nz = 0;
    for (int s = 0xC00; s < 0xE00; ++s)  // bytes 0x3000-0x3800
      if (shm[s]) pfx_nz++;
    printf("region bytes [0x3000,0x3800) nonzero i32s: %d\n", pfx_nz);
  }

  // (a) permutation check
  int bad_range = 0, dup = 0;
  std::vector<int> seen(TILE, 0);
  for (int i = 0; i < TILE; ++i) {
    if (ranks[i] < 0 || ranks[i] >= TILE) { bad_range++; continue; }
    if (seen[ranks[i]]++) dup++;
  }
  // (b) digit-nondecreasing by rank
  std::vector<int> digit_at_rank(TILE, -1);
  for (int i = 0; i < TILE; ++i)
    if (ranks[i] >= 0 && ranks[i] < TILE) digit_at_rank[ranks[i]] = digits[i];
  int order_viol = 0, prev = -1;
  for (int r = 0; r < TILE; ++r) {
    if (digit_at_rank[r] < 0) continue;
    if (digit_at_rank[r] < prev) order_viol++;
    prev = digit_at_rank[r];
  }
  // (c) per-digit rank range vs host histogram
  int hist[256] = {0};
  for (int i = 0; i < TILE; ++i) hist[in[i] & 0xFF]++;
  int range_viol = 0;
  {
    int base = 0;
    std::vector<int> lo(256, 1 << 30), hi(256, -1);
    for (int i = 0; i < TILE; ++i) {
      if (ranks[i] < 0 || ranks[i] >= TILE) continue;
      int d = digits[i];
      if (d < 0 || d > 255) continue;
      if (ranks[i] < lo[d]) lo[d] = ranks[i];
      if (ranks[i] > hi[d]) hi[d] = ranks[i];
    }
    for (int d = 0; d < 256; ++d) {
      if (hist[d] > 0 && hi[d] >= 0) {
        if (lo[d] != base || hi[d] != base + hist[d] - 1) range_viol++;
      }
      base += hist[d];
    }
  }

  printf("bad_range=%d dup=%d order_viol=%d range_viol=%d\n",
         bad_range, dup, order_viol, range_viol);
  if (bad_range == 0 && dup == 0 && order_viol == 0 && range_viol == 0) {
    printf("TEST PASSED\n");
    return 0;
  }
  // diagnostics: first items + the first duplicated/missing ranks
  for (int i = 0; i < 16; ++i)
    printf("  item %4d digit 0x%02x rank %d\n", i, digits[i], ranks[i]);
  int shown = 0;
  for (int r = 0; r < TILE && shown < 16; ++r)
    if (seen[r] != 1) { printf("  rank %4d covered %dx\n", r, seen[r]); shown++; }
  printf("FAIL\n");
  return 1;
}
