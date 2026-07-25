// tma_min — minimal Hopper TMA microbench: cp.async.bulk.tensor.2d loads a
// 32x32 f32 tile from global (via cuTensorMapEncodeTiled) into shared,
// completing on an mbarrier; block dumps the tile. Exercises SASS UTMALDG +
// SYNCS tx-arrive (A0T1) + FENCE.VIEW.ASYNC + ELECT.
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ __forceinline__ unsigned smem_u32(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__global__ void tma_kernel(const __grid_constant__ CUtensorMap tmap,
                           float* __restrict__ out, int x0, int y0) {
  __shared__ alignas(1024) float tile[32 * 32];
  __shared__ alignas(8) unsigned long long bar;
  const int tid = threadIdx.x;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(&bar)), "r"(1));
    asm volatile("fence.proxy.async.shared::cta;");
    // expect the tile bytes, then issue the bulk tensor load
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :: "r"(smem_u32(&bar)), "r"(32 * 32 * 4));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(smem_u32(tile)), "l"(&tmap), "r"(x0), "r"(y0), "r"(smem_u32(&bar)));
  }
  __syncthreads();
  unsigned done = 0;
  while (!done) {
    asm volatile("{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p;}"
                 : "=r"(done) : "r"(smem_u32(&bar)), "r"(0));
  }
  for (int i = tid; i < 32 * 32; i += blockDim.x) out[i] = tile[i];
}

int main() {
  const int W = 64, H = 64;                       // global tensor 64x64 f32
  float* g_h = new float[W * H];
  for (int i = 0; i < W * H; ++i) g_h[i] = (float)((i * 29 + 5) % 97) * 0.5f;
  float *g_d, *out_d;
  cudaMalloc(&g_d, W * H * 4); cudaMalloc(&out_d, 32 * 32 * 4);
  cudaMemcpy(g_d, g_h, W * H * 4, cudaMemcpyHostToDevice);

  CUtensorMap tmap;
  cuuint64_t dims[2] = {W, H};
  cuuint64_t strides[1] = {W * 4};        // row pitch in bytes
  cuuint32_t box[2] = {32, 32};
  cuuint32_t estr[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, g_d, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tmap encode failed %d\n", r); return 1; }

  tma_kernel<<<1, 128>>>(tmap, out_d, 16, 16);    // load box at (16,16)
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) { printf("CUDA ERROR: %s\n", cudaGetErrorString(err)); return 1; }
  float* out_h = new float[32 * 32];
  cudaMemcpy(out_h, out_d, 32 * 32 * 4, cudaMemcpyDeviceToHost);
  double s = 0; for (int i = 0; i < 32 * 32; ++i) s += out_h[i] * (i + 1);
  printf("weighted %.3f t[0]=%.2f t[33]=%.2f t[1023]=%.2f\n",
         s, out_h[0], out_h[33], out_h[1023]);
  return 0;
}
