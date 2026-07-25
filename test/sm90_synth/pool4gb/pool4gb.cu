#include <cstdio>
#include <cuda_runtime.h>
// Isolates the btree pool-addressing pattern (pool_allocator.cuh:65
// getAddressPtr: d_pool + address*32, address from atomicAdd(d_count,1)).
// Seeds the counter so node addresses land ABOVE the 4GB byte boundary,
// exercising the IMAD.WIDE.U32 64-bit pointer arithmetic. Each warp grabs a
// sequential node address, writes 32 lanes at d_pool + address*32, fences,
// reads back. A dropped high carry -> wrong address -> readback mismatch.
#define WARP_MASK 0xFFFFFFFFu
#define NODE 32u
__global__ void addr_kernel(unsigned* d_pool, unsigned* d_count, unsigned* d_err) {
  int lane = threadIdx.x & 31;
  unsigned address = 0;
  if (lane == 0) address = atomicAdd(d_count, 1u);
  address = __shfl_sync(WARP_MASK, address, 0);
  unsigned* p = d_pool + address * NODE;          // same uint32 index arith as btree
  p[lane] = address ^ (unsigned)lane;             // write sentinel
  __threadfence();
  if (p[lane] != (address ^ (unsigned)lane)) atomicAdd(d_err, 1u);
}
int main() {
  // Pool spans > 4 GB so high node addresses cross the 2^32-byte boundary.
  const unsigned START = 34000000u;               // 34M nodes * 128 B = 4.35 GB
  const unsigned NODES = 1000000u;                // cover 34.0M .. 35.0M
  size_t pool_elems = (size_t)(START + NODES) * NODE;   // ~1.12e9 elems
  size_t pool_bytes = pool_elems * sizeof(unsigned);    // ~4.48 GB
  unsigned *d_pool, *d_count, *d_err, h_err = 0;
  if (cudaMalloc(&d_pool, pool_bytes) != cudaSuccess) { printf("alloc fail\n"); return 0; }
  cudaMallocManaged(&d_count, sizeof(unsigned));
  cudaMallocManaged(&d_err, sizeof(unsigned));
  *d_count = START; *d_err = 0;
  int WGS = 256, warps_needed = NODES, threads = warps_needed * 32;
  int blocks = (threads + WGS - 1) / WGS;
  addr_kernel<<<blocks, WGS>>>(d_pool, d_count, d_err);
  cudaDeviceSynchronize();
  h_err = *d_err;
  printf("errors=%u final_count=%u expect_count=%u %s\n",
         h_err, *d_count, START + NODES,
         (h_err == 0 && *d_count == START + NODES) ? "PASS" : "FAIL");
  return 0;
}
