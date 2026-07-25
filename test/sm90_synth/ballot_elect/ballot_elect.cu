#include <cstdio>
#include <cuda_runtime.h>
// Mirrors the GpuBTree warp-cooperative insertion/allocation loop
// (map_warps.cu:324): warp elects one lane at a time via __ballot_sync +
// __ffs, the elected lane "allocates" (atomicAdd counter) and clears its
// flag; loop runs until no lane has work. Correct: exactly one allocation
// per nonzero key. A broken ballot/ffs/shfl election -> the loop fails to
// converge -> runaway allocations (btree: 212 GiB vs 0.628 GiB).
#define WARP_MASK 0xFFFFFFFFu
__global__ void elect_kernel(const unsigned* keys, unsigned* d_count) {
  int lane = threadIdx.x & 31;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned myKey = keys[gid];
  bool to_be_inserted = (myKey != 0);
  unsigned work_queue;
  unsigned iters = 0;
  while ((work_queue = __ballot_sync(WARP_MASK, to_be_inserted))) {
    int src_lane = __ffs(work_queue) - 1;
    unsigned src_key = __shfl_sync(WARP_MASK, myKey, src_lane, 32);
    if (lane == src_lane && src_key != 0) {
      atomicAdd(d_count, 1u);
      to_be_inserted = false;
    }
    if (++iters > 64u) break;   // safety cap so a broken lift can't hang the GPU
  }
}
int main() {
  const int WGS = 256, BLOCKS = 16, N = WGS * BLOCKS;
  unsigned *keys, *d_count, h_count = 0;
  cudaMallocManaged(&keys, N * sizeof(unsigned));
  cudaMallocManaged(&d_count, sizeof(unsigned));
  *d_count = 0;
  for (int i = 0; i < N; i++) keys[i] = (unsigned)(i + 1);  // all nonzero
  elect_kernel<<<BLOCKS, WGS>>>(keys, d_count);
  cudaDeviceSynchronize();
  h_count = *d_count;
  printf("count=%u expect=%d %s\n", h_count, N, (h_count == (unsigned)N) ? "PASS" : "FAIL");
  return 0;
}
