// Minimal single-UTMALDG TMA microbench: TMA-load one contiguous K-INTER tile
// of A (64x8 fp16, no swizzle) into shared memory, then store it straight back
// out to global. No HGMMA — this isolates the TMA data-movement mechanism
// (host CUtensorMap -> UTMALDG -> shared memory) so the x86 lowering can be
// verified byte-for-byte against the input. A small tile/box keeps cute from
// splitting the load into multiple UTMALDG issues.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>

using namespace cute;
using TA = half_t;

using GA_Layout = decltype(make_layout(Shape<_64,_8>{}, LayoutRight{}));
using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, Shape<_64,_8>{}));

struct SharedStorage {
  alignas(128) cute::ArrayEngine<TA, cute::cosize_v<SmemLayoutA>> sA;
  alignas(16)  uint64_t mbar[1];
};

// Concrete TMA copy type (so the kernel can be extern "C", non-templated — the
// x86 lifter/harness key on a stable symbol name, not a C++ mangled template).
static __host__ auto make_dummy_tensor() {
  return make_tensor(make_gmem_ptr((TA const*)nullptr),
                     make_layout(Shape<_64,_8>{}, LayoutRight{}));
}
using TmaA_t = decltype(make_tma_copy(SM90_TMA_LOAD{}, make_dummy_tensor(),
                                      SmemLayoutA{}));

extern "C" __global__ void
tma_simple_kernel(CUTE_GRID_CONSTANT TmaA_t const tma_a,
                  half* __restrict__ gOut, GA_Layout ga_layout) {
#if defined(__CUDA_ARCH__)
  extern __shared__ char smem_raw[];
  SharedStorage& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  Tensor sA = make_tensor(make_smem_ptr(ss.sA.begin()), SmemLayoutA{});
  Tensor mA = tma_a.get_tma_tensor(shape(ga_layout));

  auto cta_a = tma_a.get_slice(Int<0>{});
  Tensor tAgA = cta_a.partition_S(mA);
  Tensor tAsA = cta_a.partition_D(sA);

  int t = threadIdx.x;
  constexpr int kBytes = sizeof(TA) * 64 * 8;
  if (t == 0) {
    ss.mbar[0] = 0;
    cute::initialize_barrier(ss.mbar[0], 1);
    cute::set_barrier_transaction_bytes(ss.mbar[0], kBytes);
    copy(tma_a.with(ss.mbar[0]), tAgA, tAsA);
  }
  __syncthreads();
  cute::wait_barrier(ss.mbar[0], 0);
  __syncthreads();

  // Each thread copies its share of the smem tile back to global, in the
  // logical (m,k) order so the host can compare against the input directly.
  for (int i = t; i < 64 * 8; i += blockDim.x) {
    int m = i / 8, k = i % 8;
    gOut[m * 8 + k] = static_cast<half>(sA(m, k));
  }
#endif
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
    return 2;
  }
  const char* OUT = argv[1];
  const int M = 64, K = 8;
  half hA[M * K], hOut[M * K];
  for (int i = 0; i < M * K; i++) hA[i] = __float2half(((i * 37 % 13) - 6) * 0.05f);
  memset(hOut, 0, sizeof(hOut));

  half *dA, *dOut;
  cudaMalloc(&dA, sizeof(hA)); cudaMalloc(&dOut, sizeof(hOut));
  cudaMemcpy(dA, hA, sizeof(hA), cudaMemcpyHostToDevice);
  cudaMemset(dOut, 0, sizeof(hOut));

  GA_Layout ga = make_layout(Shape<_64,_8>{}, LayoutRight{});
  Tensor gmA = make_tensor(make_gmem_ptr((TA const*)dA), ga);
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gmA, SmemLayoutA{});

  int smem_bytes = (int)sizeof(SharedStorage);
  cudaFuncSetAttribute(tma_simple_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  tma_simple_kernel<<<1, 128, smem_bytes>>>(tma_a, dOut, ga);
  cudaDeviceSynchronize();
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hOut, dOut, sizeof(hOut), cudaMemcpyDeviceToHost);

  double maxabs = 0;
  for (int i = 0; i < M * K; i++) {
    double d = fabs((double)__half2float(hOut[i]) - (double)__half2float(hA[i]));
    if (d > maxabs) maxabs = d;
  }
  printf("=== TMA round-trip out[0..7] ===\n");
  for (int k = 0; k < 8; k++) printf("%.4f ", __half2float(hOut[k])); printf("\n");
  printf("max |out - in| = %.5f\n", maxabs);

  char p[256];
  snprintf(p, sizeof(p), "%s/in_A.bin", OUT);  { FILE* f = fopen(p, "wb"); fwrite(hA, sizeof(half), M * K, f); fclose(f); }
  snprintf(p, sizeof(p), "%s/ref_out.bin", OUT); { FILE* f = fopen(p, "wb"); fwrite(hOut, sizeof(half), M * K, f); fclose(f); }
  if (maxabs < 1e-3) { printf("TEST PASSED\n"); return 0; }
  printf("TEST FAILED\n");
  return 1;
}
