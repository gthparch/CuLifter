// cluster_rank — MINIMAL SM90 thread-block-cluster microbench.
// 4 CTAs launched as cluster 2x1x1. Thread 0 of each CTA records its
// %cluster_ctarank, %clusterid.x, and blockIdx.x into a global array;
// the host checks the exact expected values (rank = b%2, cluid = b/2).
// Kernel k_dyn: cluster dims supplied at launch via cudaLaunchKernelExC +
//   cudaLaunchAttributeClusterDimension (the FA3 launch path).
// Kernel k_static: identical body, static __cluster_dims__(2,1,1) attribute
//   + plain <<<>>> launch (cluster dims live in kernel metadata).
// Exercises SASS S2R SR_CLUSTER_CTARANK / cluster-id reads and the
// .explicitcluster/.reqnctapercluster PTX directive question.
#include <cstdio>
#include <cuda_runtime.h>

static __device__ __forceinline__ unsigned rd_cluster_ctarank() {
  unsigned r; asm("mov.u32 %0, %%cluster_ctarank;" : "=r"(r)); return r;
}
static __device__ __forceinline__ unsigned rd_clusterid_x() {
  unsigned r; asm("mov.u32 %0, %%clusterid.x;" : "=r"(r)); return r;
}

static __device__ __forceinline__ void record(unsigned* out) {
  if (threadIdx.x == 0) {
    unsigned b = blockIdx.x;
    out[b * 3 + 0] = rd_cluster_ctarank();
    out[b * 3 + 1] = rd_clusterid_x();
    out[b * 3 + 2] = b;
  }
}

__global__ void k_dyn(unsigned* out) { record(out); }

__global__ void __cluster_dims__(2, 1, 1) k_static(unsigned* out) {
  record(out);
}

static const int NBLK = 4;

static int check(const char* tag, unsigned* out_d) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("%s CUDA ERROR: %s\n", tag, cudaGetErrorString(err));
    printf("%s FAIL\n", tag);
    return 1;
  }
  unsigned h[NBLK * 3];
  cudaMemcpy(h, out_d, sizeof(h), cudaMemcpyDeviceToHost);
  int fail = 0;
  for (int b = 0; b < NBLK; ++b) {
    unsigned er = (unsigned)(b % 2), ec = (unsigned)(b / 2);
    printf("%s blk%d rank=%u cluid=%u bidx=%u\n", tag, b,
           h[b * 3 + 0], h[b * 3 + 1], h[b * 3 + 2]);
    if (h[b * 3 + 0] != er || h[b * 3 + 1] != ec || h[b * 3 + 2] != (unsigned)b)
      fail = 1;
  }
  printf("%s %s\n", tag, fail ? "FAIL" : "PASS");
  return fail;
}

int main() {
  unsigned* out_d;
  cudaMalloc(&out_d, NBLK * 3 * sizeof(unsigned));
  int bad = 0;

  // --- kernel 1: dynamic cluster dims via cudaLaunchKernelExC ---
  cudaMemset(out_d, 0xFF, NBLK * 3 * sizeof(unsigned));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NBLK, 1, 1);
  cfg.blockDim = dim3(32, 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  void* args[] = {&out_d};
  cudaError_t le = cudaLaunchKernelExC(&cfg, (void*)k_dyn, args);
  if (le != cudaSuccess)
    printf("k_dyn launch error: %s\n", cudaGetErrorString(le));
  bad |= check("k_dyn", out_d);

  // --- kernel 2: static __cluster_dims__(2,1,1), plain <<<>>> launch ---
  cudaMemset(out_d, 0xFF, NBLK * 3 * sizeof(unsigned));
  k_static<<<NBLK, 32>>>(out_d);
  bad |= check("k_static", out_d);

  printf("%s\n", bad ? "FAIL" : "PASS");
  return bad;
}
