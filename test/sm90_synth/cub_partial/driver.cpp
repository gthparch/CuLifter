// Host driver for the cub_partial microbenchmark (lifted Onesweep kernel,
// PARTIAL-TILE path: num_items=32 of TILE_ITEMS=1536).
//
// Setup mirrors the real DeviceRadixSort dispatch for a single pass
// (current_bit=0, num_bits=8, one block):
//   - d_lookback zeroed, d_ctrs zeroed
//   - d_bins_in[bin] (u64) = exclusive prefix of the per-digit histogram of
//     the N real keys (what DeviceRadixSortHistogram+ExclusiveSum produce)
// Expected HW behavior: keys_out[0..N) = keys stably sorted by digit
// (digit = low byte here), values_out follows keys. Nothing else written.
//
// keys_out/values_out point into the MIDDLE of big sentinel-filled buffers
// (red zones) so buggy negative/overshoot scatter indices are captured and
// reported instead of segfaulting.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

extern "C" void _ZN3cub16_V_300200_SM_9006detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiiyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_();
#define ONESWEEP _ZN3cub16_V_300200_SM_9006detail10radix_sort29DeviceRadixSortOnesweepKernelINS1_5radix10policy_hubIiiyE10Policy1000ELNS0_9SortOrderE0EiiyiiNS1_21identity_decomposer_tEEEvPT5_SB_PT3_PKSC_PT1_PKSG_PT2_PKSK_T4_iiT6_

#ifndef N_ITEMS
#define N_ITEMS 32
#endif
#define BT 384
#define TILE 6528           // BLOCK_THREADS=384 x ITEMS_PER_THREAD=17 (Policy1000)
#define RED 16384           // red zone: window [-RED, +RED) around out base
#define SENT 0x7EADBEEF     // sentinel (positive, distinguishable)

int main(int argc, char** argv) {
  unsigned seed = (argc > 1) ? (unsigned)atoi(argv[1]) : 12345u;
  const int n = (argc > 2) ? atoi(argv[2]) : N_ITEMS;

  int* keys_in = new int[TILE]();
  int* values_in = new int[TILE]();
  uint32_t x = seed ? seed : 1u;
  const bool distinct = getenv("DISTINCT_KEYS") != nullptr;
  for (int i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;  // xorshift32
    // digits in [8,200): never collide with the 0xFF default-key digit and
    // distinguishable from small structural garbage (0..7)
    keys_in[i] = distinct ? (8 + (i * 6) % 192) : (int)(8 + x % 192);
    values_in[i] = 1000 + i;
  }

  // d_bins_in: u64 exclusive prefix of digit histogram over the n real keys
  uint64_t* bins_in = new uint64_t[256]();
  {
    uint64_t hist[256] = {0};
    for (int i = 0; i < n; ++i) hist[keys_in[i] & 0xFF]++;
    uint64_t acc = 0;
    for (int d = 0; d < 256; ++d) { bins_in[d] = acc; acc += hist[d]; }
  }

  int* lookback = new int[8192]();   // 256 needed for 1 block; extra slack
  int* ctrs = new int[64]();
  uint64_t* bins_out = new uint64_t[256]();

  // out buffers with red zones; lifted kernel gets the midpoint pointer
  int* keys_buf = new int[2 * RED];
  int* values_buf = new int[2 * RED];
  for (int i = 0; i < 2 * RED; ++i) { keys_buf[i] = SENT; values_buf[i] = SENT; }
  int* keys_out = keys_buf + RED;
  int* values_out = values_buf + RED;

  launchKernel(ONESWEEP, /*grid*/ 1, /*block*/ BT,
               ArgAt{0x210}, lookback,
               ArgAt{0x218}, ctrs,
               ArgAt{0x220}, bins_out,
               ArgAt{0x228}, bins_in,
               ArgAt{0x230}, keys_out,
               ArgAt{0x238}, keys_in,
               ArgAt{0x240}, values_out,
               ArgAt{0x248}, values_in,
               ArgAt{0x250}, n,
               ArgAt{0x254}, 0,     // current_bit
               ArgAt{0x258}, 8,     // num_bits
               ArgAt{0x25c}, 0);    // identity decomposer (empty)

  // Post-kernel shared dump (gridDim=1, final CTA state persists).
  // Layout (AgentRadixSortOnesweep::TempStorage_, TILE=6528):
  //   bytes [0, 0x6600): union keys_out[6528] / values_out[6528] / rank tmp
  //     - after ScatterValuesShared this region holds values by rank
  //   bytes [0x6600, 0x6e00): union global_offsets[256] (u64) / block_idx
  if (getenv("DUMP_SHM")) {
    const int32_t* shm = shared_mem;
    printf("shared values-by-rank [0..47]:\n");
    for (int r = 0; r < 48; r += 8)
      printf("  r%2d: %d %d %d %d %d %d %d %d\n", r, shm[r], shm[r+1], shm[r+2],
             shm[r+3], shm[r+4], shm[r+5], shm[r+6], shm[r+7]);
    printf("global_offsets (i64) digits 0..15:\n");
    for (int d = 0; d < 16; ++d) {
      long long v;
      memcpy(&v, (const char*)shm + 0x6600 + 8u * d, 8);
      printf("  d%d=%lld(0x%llx)", d, v, (unsigned long long)v);
    }
    printf("\n");
    // full scan: nonzero extents over the whole wrapper shared segment
    printf("nonzero shared extents (i32 idx [start,end) : first vals):\n");
    int i = 0, total = 49152;
    while (i < total) {
      if (shm[i] == 0) { i++; continue; }
      int s = i;
      while (i < total && shm[i] != 0) i++;
      printf("  [0x%x,0x%x) bytes [0x%x,0x%x) n=%d :", s, i, 4*s, 4*i, i-s);
      for (int k = s; k < s + 6 && k < i; ++k) printf(" 0x%x", (unsigned)shm[k]);
      printf("\n");
    }
    // where did values_in (1000+i) and twiddled keys (0x800000xx) land?
    printf("value-class locations:\n");
    for (int k = 0; k < total; ++k) {
      unsigned v = (unsigned)shm[k];
      if ((v >= 1000 && v < 1000 + 64) || (v & 0xFFFFFF00u) == 0x80000000u)
        printf("  byte 0x%05x = 0x%08x (%d)\n", 4 * k, v, shm[k]);
    }
  }

  // CPU reference: stable sort (key,value) pairs by digit (= low byte)
  std::vector<std::pair<int,int>> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = {keys_in[i], values_in[i]};
  std::stable_sort(ref.begin(), ref.end(),
                   [](const std::pair<int,int>& a, const std::pair<int,int>& b) {
                     return (a.first & 0xFF) < (b.first & 0xFF);
                   });

  int key_mismatch = 0, val_mismatch = 0, oob_writes = 0;
  for (int i = 0; i < n; ++i) {
    if (keys_out[i] != ref[i].first) {
      if (key_mismatch < 12)
        printf("  keys_out[%d] = %d (0x%08x), want %d\n", i, keys_out[i],
               (unsigned)keys_out[i], ref[i].first);
      key_mismatch++;
    }
    if (values_out[i] != ref[i].second) {
      if (val_mismatch < 12)
        printf("  values_out[%d] = %d, want %d\n", i, values_out[i], ref[i].second);
      val_mismatch++;
    }
  }
  for (int i = 0; i < 2 * RED; ++i) {
    int rel = i - RED;
    if (rel >= 0 && rel < n) continue;  // expected write window
    if (keys_buf[i] != SENT) {
      if (oob_writes < 16)
        printf("  OOB key write at idx %d: %d (0x%08x)\n", rel, keys_buf[i],
               (unsigned)keys_buf[i]);
      oob_writes++;
    }
    if (values_buf[i] != SENT) {
      if (oob_writes < 16)
        printf("  OOB value write at idx %d: %d\n", rel, values_buf[i]);
      oob_writes++;
    }
  }

  printf("n=%d key_mismatch=%d val_mismatch=%d oob_writes=%d\n",
         n, key_mismatch, val_mismatch, oob_writes);
  if (key_mismatch == 0 && val_mismatch == 0 && oob_writes == 0) {
    printf("TEST PASSED\n");
    return 0;
  }
  printf("FAIL\n");
  return 1;
}
