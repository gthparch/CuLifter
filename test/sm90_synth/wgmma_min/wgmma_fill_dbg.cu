// fill-loop isolation: same smem fill as wgmma_min, dump smem to global.
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
__global__ void fill_kernel(const __nv_bfloat16* __restrict__ A,
                            const __nv_bfloat16* __restrict__ B,
                            float* __restrict__ out) {
  __shared__ alignas(128) __nv_bfloat16 sA[64 * 16];
  __shared__ alignas(128) __nv_bfloat16 sB[64 * 16];
  const int tid = threadIdx.x;
  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    int i = idx / 16, k = idx % 16;
    int off = (i / 8) * 64 + (i % 8) * 8 + (k % 8) + (k / 8) * 512;
    sA[off] = A[i * 16 + k];
    sB[off] = B[i * 16 + k];
  }
  __syncthreads();
  for (int idx = tid; idx < 64 * 16; idx += blockDim.x) {
    out[idx] = __bfloat162float(sA[idx]);
    out[64 * 16 + idx] = __bfloat162float(sB[idx]);
  }
}
int main() {
  const int MK = 64 * 16;
  __nv_bfloat16 *A_h = new __nv_bfloat16[MK], *B_h = new __nv_bfloat16[MK];
  for (int i = 0; i < MK; ++i) A_h[i] = __float2bfloat16((float)((i * 13 + 7) % 17) * 0.125f);
  for (int i = 0; i < MK; ++i) B_h[i] = __float2bfloat16((float)((i * 5 + 3) % 19) * 0.0625f);
  __nv_bfloat16 *A_d, *B_d; float* out_d;
  cudaMalloc(&A_d, MK * 2); cudaMalloc(&B_d, MK * 2); cudaMalloc(&out_d, 2 * MK * 4);
  cudaMemcpy(A_d, A_h, MK * 2, cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, MK * 2, cudaMemcpyHostToDevice);
  fill_kernel<<<1, 128>>>(A_d, B_d, out_d);
  if (cudaDeviceSynchronize() != cudaSuccess) { printf("CUDA ERROR\n"); return 1; }
  float* out_h = new float[2 * MK];
  cudaMemcpy(out_h, out_d, 2 * MK * 4, cudaMemcpyDeviceToHost);
  double s = 0; for (int i = 0; i < 2 * MK; ++i) s += out_h[i] * (i + 1);
  printf("weighted checksum %.4f\n", s);
  printf("sA[0]=%.3f sA[100]=%.3f sB[512]=%.3f sB[1023]=%.3f\n",
         out_h[0], out_h[100], out_h[MK + 512], out_h[MK + 1023]);
  return 0;
}
