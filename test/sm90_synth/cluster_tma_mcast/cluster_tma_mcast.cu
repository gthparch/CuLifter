// cluster_tma_mcast — Hopper TMA MULTICAST microbench. Cluster 2x1x1 launched
// via cudaLaunchKernelExC (like FA3). Both CTAs init their own mbarrier and
// post arrive.expect_tx for the full tile; after a cluster barrier, the
// elected thread of cluster rank 0 issues ONE
// cp.async.bulk.tensor.2d...multicast::cluster with ctaMask=0b11 — the TMA
// unit multicasts the 32x32 f32 tile into BOTH CTAs' shared memory and
// delivers the complete_tx byte count to BOTH mbarriers. Each CTA waits on
// its own barrier and dumps its tile to a per-rank global slot; host checks
// both copies against the source tensor. Exercises SASS UTMALDG (multicast
// form), SYNCS tx-arrive, FENCE.VIEW.ASYNC, BAR cluster arrive/wait, ELECT.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define TW 32
#define TH 32
#define TILE (TW * TH)

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void mcast_kernel(const __grid_constant__ CUtensorMap tmap,
                             float* __restrict__ out, int x0, int y0) {
  __shared__ alignas(1024) float tile[TILE];
  __shared__ alignas(8) unsigned long long bar;
  const int tid = threadIdx.x;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_u32(&bar)), "r"(1));
    asm volatile("fence.proxy.async.shared::cta;");
    // every CTA in the mask expects the full tile byte count on ITS barrier —
    // the multicast copy delivers the complete_tx to each of them
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(&bar)), "r"(TILE * 4));
  }
  // cluster-wide barrier: all mbarriers initialized + expect_tx posted before
  // rank 0 issues the multicast
  asm volatile("barrier.cluster.arrive.aligned;");
  asm volatile("barrier.cluster.wait.aligned;");

  if (rank == 0 && tid == 0) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3}], [%4], %5;"
        :: "r"(smem_u32(tile)), "l"(&tmap), "r"(x0), "r"(y0),
           "r"(smem_u32(&bar)), "h"((unsigned short)0x3));
  }
  // each CTA spins on ITS OWN barrier (phase 0)
  unsigned done = 0;
  while (!done) {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done) : "r"(smem_u32(&bar)), "r"(0));
  }
  __syncthreads();
  float* dst = out + rank * TILE;
  for (int i = tid; i < TILE; i += blockDim.x) dst[i] = tile[i];
  // keep both CTAs alive until everyone is done (smem stays valid for TMA)
  asm volatile("barrier.cluster.arrive.aligned;");
  asm volatile("barrier.cluster.wait.aligned;");
}

int main() {
  const int W = 64, H = 64;  // global tensor 64x64 f32
  const int x0 = 16, y0 = 16;
  float* g_h = new float[W * H];
  for (int i = 0; i < W * H; ++i) g_h[i] = (float)((i * 29 + 5) % 97) * 0.5f;
  float *g_d, *out_d;
  cudaMalloc(&g_d, W * H * 4);
  cudaMalloc(&out_d, 2 * TILE * 4);
  cudaMemcpy(g_d, g_h, W * H * 4, cudaMemcpyHostToDevice);
  cudaMemset(out_d, 0xEE, 2 * TILE * 4);

  CUtensorMap tmap;
  cuuint64_t dims[2] = {W, H};
  cuuint64_t strides[1] = {W * 4};  // row pitch in bytes
  cuuint32_t box[2] = {TW, TH};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, g_d, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tmap encode failed %d\nFAIL\n", r); return 1; }

  // explicit cluster launch, 2x1x1 — same API path FA3 uses
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  int xa = x0, ya = y0;
  void* args[] = {(void*)&tmap, (void*)&out_d, (void*)&xa, (void*)&ya};
  cudaError_t lerr = cudaLaunchKernelExC(&cfg, (void*)mcast_kernel, args);
  if (lerr != cudaSuccess) {
    printf("LAUNCH ERROR: %s\nFAIL\n", cudaGetErrorString(lerr));
    return 1;
  }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA ERROR: %s\nFAIL\n", cudaGetErrorString(err));
    return 1;
  }

  float* out_h = new float[2 * TILE];
  cudaMemcpy(out_h, out_d, 2 * TILE * 4, cudaMemcpyDeviceToHost);
  bool ok = true;
  for (int cta = 0; cta < 2; ++cta) {
    double s = 0;
    int bad = 0, badi = -1;
    for (int j = 0; j < TH; ++j)
      for (int i = 0; i < TW; ++i) {
        float got = out_h[cta * TILE + j * TW + i];
        float want = g_h[(y0 + j) * W + (x0 + i)];
        s += got * (j * TW + i + 1);
        if (got != want && ++bad == 1) badi = j * TW + i;
      }
    printf("cta%d weighted %.3f bad=%d first_bad=%d t[0]=%.2f t[33]=%.2f t[1023]=%.2f\n",
           cta, s, bad, badi, out_h[cta * TILE + 0], out_h[cta * TILE + 33],
           out_h[cta * TILE + 1023]);
    if (bad) ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
