// cluster_sync_bar — Hopper cluster-wide barrier + large dynamic smem microbench.
// Mirrors FlashAttention-3's launch shape: cudaLaunchKernelExC with a 2x1x1
// cluster attribute and a LARGE (112KB) dynamic shared-memory allocation raised
// via cudaFuncAttributeMaxDynamicSharedMemorySize.
// Each CTA fills its OWN dynamic smem with rank-dependent hashed values,
// executes a cluster-wide barrier (barrier.cluster.arrive/wait -> BAR.CLUSTER
// SASS), then reads back its OWN smem (reversed indices, cross-thread within
// the CTA) and writes per-thread integer checksums to global. No remote
// (mapa/dsmem) access — that is cluster_dsmem's job. Host recomputes exactly.
#include <cstdio>
#include <cuda_runtime.h>

#define SMEM_BYTES (112 * 1024)                    // 112KB dynamic smem per CTA
#define SMEM_WORDS (SMEM_BYTES / 4)                // 28672 u32 elements
#define THREADS 128
#define BLOCKS 4                                   // 2 clusters of 2 CTAs

__host__ __device__ __forceinline__ unsigned hashval(unsigned rank, unsigned i) {
  unsigned v = rank * 1315423911u + i * 2654435761u;
  v ^= v >> 15;
  return v;
}

__global__ void cluster_kernel(unsigned* __restrict__ out) {
  extern __shared__ unsigned smem[];
  const unsigned rank = blockIdx.x;        // rank-dependent fill (per CTA)
  const unsigned tid = threadIdx.x;

  for (unsigned i = tid; i < SMEM_WORDS; i += blockDim.x)
    smem[i] = hashval(rank, i);
  __syncthreads();

  // Cluster-wide barrier: every thread of every CTA in the cluster arrives,
  // then waits until the whole cluster has arrived.
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");

  // Read back OWN smem, reversed so each thread consumes values written by a
  // different thread of its CTA (correct only if smem is addressed right).
  unsigned acc = 0;
  for (unsigned i = tid; i < SMEM_WORDS; i += blockDim.x)
    acc += smem[SMEM_WORDS - 1 - i];
  out[blockIdx.x * blockDim.x + tid] = acc;
}

int main() {
  unsigned* out_d;
  unsigned out_h[BLOCKS * THREADS];
  cudaMalloc(&out_d, sizeof(out_h));
  cudaMemset(out_d, 0, sizeof(out_h));

  cudaError_t err = cudaFuncSetAttribute(
      cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_BYTES);
  if (err != cudaSuccess) { printf("attr FAIL: %s\n", cudaGetErrorString(err)); return 1; }

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(BLOCKS, 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM_BYTES;
  cfg.stream = 0;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  void* args[] = {&out_d};
  err = cudaLaunchKernelExC(&cfg, (const void*)cluster_kernel, args);
  if (err != cudaSuccess) { printf("launch FAIL: %s\n", cudaGetErrorString(err)); return 1; }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("sync FAIL: %s\n", cudaGetErrorString(err)); return 1; }
  cudaMemcpy(out_h, out_d, sizeof(out_h), cudaMemcpyDeviceToHost);

  int bad = 0;
  unsigned long long total = 0;
  for (unsigned b = 0; b < BLOCKS; ++b) {
    for (unsigned t = 0; t < THREADS; ++t) {
      unsigned exp = 0;
      for (unsigned i = t; i < SMEM_WORDS; i += THREADS)
        exp += hashval(b, SMEM_WORDS - 1 - i);
      unsigned got = out_h[b * THREADS + t];
      total += got;
      if (got != exp) {
        if (bad < 4)
          printf("MISMATCH b=%u t=%u got=%08x exp=%08x\n", b, t, got, exp);
        ++bad;
      }
    }
  }
  printf("checksum %llx\n", total);
  printf("o[0]=%08x o[127]=%08x o[128]=%08x o[511]=%08x\n",
         out_h[0], out_h[127], out_h[128], out_h[BLOCKS * THREADS - 1]);
  if (bad == 0) printf("PASS\n");
  else printf("FAIL (%d mismatches)\n", bad);
  return bad != 0;
}
