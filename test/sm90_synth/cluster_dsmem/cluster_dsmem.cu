// cluster_dsmem — Hopper distributed-shared-memory (DSMEM) microbench.
// Cluster 2x1x1, 2 clusters (4 CTAs total). Each CTA fills ITS OWN
// __shared__ array with rank-dependent values, cluster.sync(), then reads
// the PEER CTA's shared array through cg::cluster_group::map_shared_rank
// (SASS: MAPA + LDS to the remote DSMEM window) and stores what it read
// to global memory. Host verifies the cross-CTA exchange with exact values.
// Launched via cudaLaunchKernelExC + cudaLaunchAttributeClusterDimension,
// the same explicit cluster API FlashAttention-3 uses.
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define THREADS 64
#define BLOCKS 4  // cluster size 2 -> 2 clusters

__global__ void dsmem_kernel(const int* __restrict__ in, int* __restrict__ out) {
  __shared__ int mine[THREADS];
  cg::cluster_group cluster = cg::this_cluster();
  const unsigned tid = threadIdx.x;
  const unsigned my_rank = cluster.block_rank();
  const unsigned peer_rank = my_rank ^ 1u;

  // Rank-dependent fill of MY OWN shared array (globally unique per CTA).
  mine[tid] = in[tid] + (int)(blockIdx.x + 1) * 100000 + (int)my_rank * 1000;
  cluster.sync();

  // Read the PEER CTA's shared array via the DSMEM window.
  int* peer = cluster.map_shared_rank(mine, peer_rank);
  int got = peer[(tid + 1u) % THREADS];  // non-identity index to defeat forwarding
  out[blockIdx.x * THREADS + tid] = got;

  // Keep this CTA's smem alive until the peer has finished reading it.
  cluster.sync();
}

int main() {
  const int n = BLOCKS * THREADS;
  int in_h[THREADS], out_h[n];
  for (int i = 0; i < THREADS; ++i) in_h[i] = (i * 7 + 3) % 50;

  int *in_d, *out_d;
  cudaMalloc(&in_d, THREADS * sizeof(int));
  cudaMalloc(&out_d, n * sizeof(int));
  cudaMemcpy(in_d, in_h, THREADS * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(out_d, 0xFF, n * sizeof(int));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(BLOCKS, 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeClusterDimension;
  attrs[0].val.clusterDim.x = 2;
  attrs[0].val.clusterDim.y = 1;
  attrs[0].val.clusterDim.z = 1;
  cfg.attrs = attrs;
  cfg.numAttrs = 1;

  cudaError_t err = cudaLaunchKernelEx(&cfg, dsmem_kernel,
                                       (const int*)in_d, out_d);
  if (err != cudaSuccess) { printf("LAUNCH ERROR: %s\nFAIL\n", cudaGetErrorString(err)); return 1; }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\nFAIL\n", cudaGetErrorString(err)); return 1; }
  cudaMemcpy(out_h, out_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  // Expected: CTA b reads from peer CTA pb = b^1 (rank pb%2), element (t+1)%THREADS.
  int bad = 0;
  long long sum = 0;
  for (int b = 0; b < BLOCKS; ++b) {
    int pb = b ^ 1;
    for (int t = 0; t < THREADS; ++t) {
      int src = (t + 1) % THREADS;
      int expect = in_h[src] + (pb + 1) * 100000 + (pb % 2) * 1000;
      int got = out_h[b * THREADS + t];
      sum += got;
      if (got != expect) {
        if (bad < 5) printf("MISMATCH b=%d t=%d got=%d expect=%d\n", b, t, got, expect);
        ++bad;
      }
    }
  }
  printf("checksum %lld\n", sum);
  printf("o[0]=%d o[63]=%d o[64]=%d o[255]=%d\n", out_h[0], out_h[63], out_h[64], out_h[255]);
  printf(bad == 0 ? "PASS\n" : "FAIL\n");
  return bad == 0 ? 0 : 1;
}
