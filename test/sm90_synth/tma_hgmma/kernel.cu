// Standalone sm_90a TMA -> HGMMA microbench (mirrors FlashAttention operand load).
//
// A genuine TMA bulk load (cp.async.bulk.tensor / UTMALDG) stages a 64x16 K-major
// fp16 tile of A and a 16x16 K-major tile of B from global memory into shared
// memory using a host-encoded CUtensorMap (cuTensorMapEncodeTiled, via cute's
// make_tma_copy), then a single warpgroup HGMMA.64x16x16.F32 computes
// D[64x16] = A[64x16] * B[16x16]^T (fp16 in, fp32 accum).
//
// This is the exact shape the prior verified wgmma_ss_fa microbench used, but the
// operands now arrive via TMA instead of per-thread element copies -- so the SASS
// contains real UTMALDG ops whose descriptor is the opaque CUtensorMap in the
// kernel param/const bank (NOT in the SASS). This is the single blocker the x86
// lifter must close for FlashAttention.
//
// SMEM layout: GMMA::Layout_K_INTER_Atom (NO swizzle) -- the swizzle-free K-major
// canonical layout the existing _try_wgmma_descriptor_product already decodes. A
// follow-on microbench adds the 128B-swizzled variant.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

using namespace cute;

using TA = half_t;
using TB = half_t;

// K-major (LayoutRight) global tensors. A is MxK = 64x16, B is NxK = 16x16.
using GA_Layout = decltype(make_layout(Shape<_64,_16>{}, LayoutRight{}));
using GB_Layout = decltype(make_layout(Shape<_16,_16>{}, LayoutRight{}));

// No-swizzle K-INTER smem layouts, tiled to the full tile (cosize == elem count).
using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, Shape<_64,_16>{}));
using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TB>{}, Shape<_16,_16>{}));

struct SharedStorage {
  alignas(128) cute::ArrayEngine<TA, cute::cosize_v<SmemLayoutA>> sA;
  alignas(128) cute::ArrayEngine<TB, cute::cosize_v<SmemLayoutB>> sB;
  alignas(16)  uint64_t mbar[1];
};

// Concrete TMA copy types so the kernel can be extern "C" (stable symbol name).
static __host__ auto _dummy_A() {
  return make_tensor(make_gmem_ptr((TA const*)nullptr),
                     make_layout(Shape<_64,_16>{}, LayoutRight{}));
}
static __host__ auto _dummy_B() {
  return make_tensor(make_gmem_ptr((TB const*)nullptr),
                     make_layout(Shape<_16,_16>{}, LayoutRight{}));
}
using TmaA_t = decltype(make_tma_copy(SM90_TMA_LOAD{}, _dummy_A(), SmemLayoutA{}));
using TmaB_t = decltype(make_tma_copy(SM90_TMA_LOAD{}, _dummy_B(), SmemLayoutB{}));

extern "C" __global__ void
tma_hgmma_kernel(CUTE_GRID_CONSTANT TmaA_t const tma_a,
                 CUTE_GRID_CONSTANT TmaB_t const tma_b,
                 float* __restrict__ gD,
                 GA_Layout ga_layout, GB_Layout gb_layout) {
#if defined(__CUDA_ARCH__)
  extern __shared__ char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);

  Tensor sA = make_tensor(make_smem_ptr(ss.sA.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(ss.sB.begin()), SmemLayoutB{});

  Tensor mA = tma_a.get_tma_tensor(shape(ga_layout));
  Tensor mB = tma_b.get_tma_tensor(shape(gb_layout));

  // Single-CTA, single tile: partition source/dest for the TMA.
  auto cta_a = tma_a.get_slice(Int<0>{});
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tAgA = cta_a.partition_S(mA);
  Tensor tAsA = cta_a.partition_D(sA);
  Tensor tBgB = cta_b.partition_S(mB);
  Tensor tBsB = cta_b.partition_D(sB);

  int t = threadIdx.x;
  constexpr int kBytesA = sizeof(TA) * 64 * 16;
  constexpr int kBytesB = sizeof(TB) * 16 * 16;

  if (t == 0) {
    ss.mbar[0] = 0;
    cute::initialize_barrier(ss.mbar[0], 1);
    cute::set_barrier_transaction_bytes(ss.mbar[0], kBytesA + kBytesB);
    copy(tma_a.with(ss.mbar[0]), tAgA, tAsA);
    copy(tma_b.with(ss.mbar[0]), tBgB, tBsB);
  }
  __syncthreads();
  cute::wait_barrier(ss.mbar[0], 0);
  __syncthreads();

  // Single-tile GMMA, SS form (A and B from smem descriptors).
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x16x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(t);

  Tensor tCsA = thr_mma.partition_A(sA);
  Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor accum = partition_fragment_C(tiled_mma, Shape<_64,_16>{});
  clear(accum);

  warpgroup_fence_operand(accum);
  warpgroup_arrive();
  gemm(tiled_mma, tCrA, tCrB, accum);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

  Tensor mD = make_tensor(make_gmem_ptr(gD), make_layout(Shape<_64,_16>{}, LayoutRight{}));
  Tensor tCgD = thr_mma.partition_C(mD);
  for (int i = 0; i < size(accum); ++i) tCgD(i) = accum(i);
#endif
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
    return 2;
  }
  const char* OUT = argv[1];
  const int M = 64, K = 16, N = 16;
  half hA[M * K], hB[N * K];
  float hD[M * N];
  for (int i = 0; i < M * K; i++) hA[i] = __float2half(((i * 37 % 13) - 6) * 0.05f);
  for (int i = 0; i < N * K; i++) hB[i] = __float2half(((i * 29 % 11) - 5) * 0.05f);
  memset(hD, 0, sizeof(hD));

  half *dA, *dB; float *dD;
  cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dB, sizeof(hB)); cudaMalloc(&dD, sizeof(hD));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, sizeof(hB), cudaMemcpyHostToDevice);
  cudaMemset(dD, 0, sizeof(hD));

  GA_Layout ga_layout = make_layout(Shape<_64,_16>{}, LayoutRight{});
  GB_Layout gb_layout = make_layout(Shape<_16,_16>{}, LayoutRight{});

  Tensor gmA = make_tensor(make_gmem_ptr((TA const*)dA), ga_layout);
  Tensor gmB = make_tensor(make_gmem_ptr((TB const*)dB), gb_layout);

  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gmA, SmemLayoutA{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gmB, SmemLayoutB{});

  int smem_bytes = (int)sizeof(SharedStorage);
  cudaFuncSetAttribute(tma_hgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  tma_hgmma_kernel<<<1, 128, smem_bytes>>>(tma_a, tma_b, dD, ga_layout, gb_layout);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hD, dD, sizeof(hD), cudaMemcpyDeviceToHost);

  double maxabs = 0;
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    double acc = 0;
    for (int k = 0; k < K; k++)
      acc += (double)__half2float(hA[m * K + k]) * (double)__half2float(hB[n * K + k]);
    double diff = fabs(acc - (double)hD[m * N + n]);
    if (diff > maxabs) maxabs = diff;
  }
  printf("=== native TMA->HGMMA D row0 ===\n");
  for (int n = 0; n < N; n++) printf("%.4f ", hD[n]); printf("\n");
  printf("max |TMA-HGMMA - cpu_fp32_ref| = %.5f\n", maxabs);

  char p[256];
  snprintf(p, sizeof(p), "%s/ref_D.bin", OUT); { FILE* f = fopen(p, "wb"); fwrite(hD, sizeof(float), M * N, f); fclose(f); }
  snprintf(p, sizeof(p), "%s/in_A.bin", OUT);  { FILE* f = fopen(p, "wb"); fwrite(hA, sizeof(half), M * K, f); fclose(f); }
  snprintf(p, sizeof(p), "%s/in_B.bin", OUT);  { FILE* f = fopen(p, "wb"); fwrite(hB, sizeof(half), N * K, f); fclose(f); }
  if (maxabs < 1e-2) { printf("TEST PASSED\n"); return 0; }
  printf("TEST FAILED\n");
  return 1;
}
