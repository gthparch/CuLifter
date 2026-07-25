// stsm_min — stmatrix (SASS STSM.16.M88.4) microbench: each warp stores a
// 16x16 b16 tile from per-lane fragments via stmatrix.x4, then the block
// dumps shared linearly. Vanilla-vs-lifted stdout must match bitwise.
#include <cstdio>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void stsm_kernel(const unsigned* __restrict__ in,
                            unsigned* __restrict__ out) {
  __shared__ alignas(128) unsigned short tile[2 * 16 * 16];  // 2 warps
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  // 4 fragment words per lane, deterministic from input
  unsigned f0 = in[tid * 4 + 0], f1 = in[tid * 4 + 1],
           f2 = in[tid * 4 + 2], f3 = in[tid * 4 + 3];
  // stmatrix.x4: lanes 0..7 address rows of m0, 8..15 m1, 16..23 m2, 24..31 m3
  // target: warp's 16x16 tile; matrix m covers rows (m%2)*8.., cols (m/2)*8..
  int m = lane / 8, r = lane % 8;
  int row = (m % 2) * 8 + r, col = (m / 2) * 8;
  unsigned addr = smem_u32(&tile[warp * 256 + row * 16 + col]);
  asm volatile("stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1,%2,%3,%4};"
               :: "r"(addr), "r"(f0), "r"(f1), "r"(f2), "r"(f3));
  __syncthreads();
  const unsigned* t32 = reinterpret_cast<const unsigned*>(tile);
  out[tid * 4 + 0] = t32[tid * 4 + 0];
  out[tid * 4 + 1] = t32[tid * 4 + 1];
  out[tid * 4 + 2] = t32[tid * 4 + 2];
  out[tid * 4 + 3] = t32[tid * 4 + 3];
}

int main() {
  const int N = 64 * 4;
  unsigned *in_h = new unsigned[N], *out_h = new unsigned[N];
  for (int i = 0; i < N; ++i) in_h[i] = (unsigned)(i * 2654435761u + 12345u);
  unsigned *in_d, *out_d;
  cudaMalloc(&in_d, N * 4); cudaMalloc(&out_d, N * 4);
  cudaMemcpy(in_d, in_h, N * 4, cudaMemcpyHostToDevice);
  stsm_kernel<<<1, 64>>>(in_d, out_d);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA ERROR\n"); return 1; }
  cudaMemcpy(out_h, out_d, N * 4, cudaMemcpyDeviceToHost);
  unsigned long long h = 1469598103934665603ull;
  for (int i = 0; i < N; ++i) { h ^= out_h[i]; h *= 1099511628211ull; }
  printf("fnv %llx out[0]=%x out[100]=%x out[255]=%x\n", h, out_h[0], out_h[100], out_h[255]);
  return 0;
}
