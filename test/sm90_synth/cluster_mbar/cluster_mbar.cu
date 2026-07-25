// cluster_mbar — Hopper cluster-scope mbarrier arrival microbench.
// Cluster 2x1x1 (explicit cudaLaunchKernelExC + cudaLaunchAttributeClusterDimension,
// like FlashAttention-3). CTA rank 0 owns an mbarrier in its shared memory;
// every thread of CTA rank 1 maps rank 0's shared addresses via mapa
// (cluster.map_shared_rank), stores one mailbox word remotely
// (st.shared::cluster), then arrives REMOTELY on rank 0's barrier with
// mbarrier.arrive.release.cluster.shared::cluster — the SYNCS.ARRIVE-on-
// remote-CTA-address construct FA3's producer/consumer handoff needs.
// CTA rank 0 spins on mbarrier.try_wait.parity.acquire.cluster, then copies
// the received mailbox plus a success flag to global; host verifies.
// Deterministic stdout: per-element values then PASS/FAIL.
#include <cstdio>
#include <cuda_runtime.h>

#define THREADS 32

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void cluster_mbar_kernel(const int* __restrict__ in,
                                    int* __restrict__ out) {
  __shared__ alignas(8) unsigned long long bar;
  __shared__ int mailbox[THREADS];
  const int tid = threadIdx.x;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned bar_addr = smem_u32(&bar);
  const unsigned mb_addr  = smem_u32(mailbox);

  if (rank == 0) {
    mailbox[tid] = -1;  // sentinel, must be overwritten remotely by rank 1
    if (tid == 0) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                   :: "r"(bar_addr), "r"((unsigned)THREADS));
      asm volatile("fence.mbarrier_init.release.cluster;");
    }
  }
  __syncthreads();
  // cluster.sync(): rank 1 must not touch the barrier before init is visible.
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;");

  if (rank == 1) {
    // Map rank 0's mailbox slot and barrier into this CTA's cluster view.
    unsigned rmb, rbar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(rmb) : "r"(mb_addr + tid * 4u));
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(rbar) : "r"(bar_addr));
    int val = in[tid] * 3 + 7;
    asm volatile("st.shared::cluster.u32 [%0], %1;" :: "r"(rmb), "r"(val));
    // Remote arrival on CTA 0's barrier (release orders this thread's store).
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0], 1;"
                 :: "r"(rbar));
  } else {
    if (tid == 0) {
      unsigned done = 0, parity = 0;
      while (!done) {
        asm volatile("{.reg .pred p; "
                     "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2; "
                     "selp.u32 %0, 1, 0, p;}"
                     : "=r"(done) : "r"(bar_addr), "r"(parity));
      }
    }
    __syncthreads();
    out[tid] = mailbox[tid];
    if (tid == 0) out[THREADS] = 12345;  // success flag
  }
  // Keep both CTAs alive until the handoff is complete.
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;");
}

int main() {
  const int n = THREADS;
  int in_h[THREADS], out_h[THREADS + 1];
  for (int i = 0; i < n; ++i) in_h[i] = (i * 13 + 5) % 97;
  int *in_d, *out_d;
  cudaMalloc(&in_d, n * sizeof(int));
  cudaMalloc(&out_d, (n + 1) * sizeof(int));
  cudaMemcpy(in_d, in_h, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(out_d, 0, (n + 1) * sizeof(int));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(2, 1, 1);   // one cluster of 2 CTAs
  cfg.blockDim = dim3(THREADS, 1, 1);
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 2;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  void* args[] = { &in_d, &out_d };
  cudaError_t lerr = cudaLaunchKernelExC(&cfg, (void*)cluster_mbar_kernel, args);
  if (lerr != cudaSuccess) { printf("LAUNCH ERROR: %s\nFAIL\n", cudaGetErrorString(lerr)); return 1; }
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\nFAIL\n", cudaGetErrorString(err)); return 1; }
  cudaMemcpy(out_h, out_d, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

  int bad = 0;
  for (int i = 0; i < n; ++i) {
    int exp = in_h[i] * 3 + 7;
    printf("o[%02d]=%d exp=%d\n", i, out_h[i], exp);
    if (out_h[i] != exp) ++bad;
  }
  printf("flag=%d\n", out_h[n]);
  if (out_h[n] != 12345) ++bad;
  printf(bad == 0 ? "PASS\n" : "FAIL\n");
  return bad != 0;
}
