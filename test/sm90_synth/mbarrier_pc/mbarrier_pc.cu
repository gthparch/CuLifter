// mbarrier_pc — Hopper mbarrier producer/consumer microbench.
// Exercises the SYNCS SASS family (EXCH=init, ARRIVE, PHASECHK[.TRYWAIT])
// that FlashAttention-3's warp-specialized ring buffer is built on:
// every thread fills a shared tile, arrives on an mbarrier, spins on
// try_wait.parity, then reads the tile written by the "opposite" thread.
// Deterministic checksum printed; vanilla-vs-lifted must match bit-for-bit.
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void pc_kernel(const float* __restrict__ in, float* __restrict__ out,
                          int n, int iters) {
  __shared__ alignas(8) unsigned long long bar;
  __shared__ float tile[256];
  const unsigned bar_addr = smem_u32(&bar);
  const int tid = threadIdx.x;

  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(bar_addr), "r"(blockDim.x));
  }
  __syncthreads();

  float acc = 0.f;
  unsigned parity = 0;
  for (int it = 0; it < iters; ++it) {
    tile[tid] = in[(it * 256 + tid) % n] * (float)(it + 1);
    unsigned long long state;
    asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];"
                 : "=l"(state) : "r"(bar_addr));
    unsigned done = 0;
    while (!done) {
      asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                   : "=r"(done) : "r"(bar_addr), "r"(parity));
    }
    parity ^= 1u;
    acc += tile[255 - tid];      // cross-thread read: only correct if the barrier works
  }
  out[blockIdx.x * blockDim.x + tid] = acc;
}

int main() {
  const int n = 1024, iters = 8, threads = 256, blocks = 2;
  float *in_h = new float[n], *out_h = new float[blocks * threads];
  for (int i = 0; i < n; ++i) in_h[i] = (float)((i * 37 + 11) % 101) * 0.25f;
  float *in_d, *out_d;
  cudaMalloc(&in_d, n * sizeof(float));
  cudaMalloc(&out_d, blocks * threads * sizeof(float));
  cudaMemcpy(in_d, in_h, n * sizeof(float), cudaMemcpyHostToDevice);
  pc_kernel<<<blocks, threads>>>(in_d, out_d, n, iters);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  cudaMemcpy(out_h, out_d, blocks * threads * sizeof(float), cudaMemcpyDeviceToHost);
  double sum = 0; for (int i = 0; i < blocks * threads; ++i) sum += out_h[i];
  printf("checksum %.6f\n", sum);
  printf("o[0]=%.4f o[255]=%.4f o[256]=%.4f o[511]=%.4f\n",
         out_h[0], out_h[255], out_h[256], out_h[511]);
  return 0;
}
