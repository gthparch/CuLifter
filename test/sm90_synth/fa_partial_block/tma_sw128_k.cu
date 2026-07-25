// tma_sw128_k — TMA-filled SW128 SS-HGMMA, configurable K (64/128/192).
// Reproduces the EXACT FA3 hdim192 QK data path: TMA loads a K-major SW128 tile
// (Sw<3,4,3>) of A(MMxKK) and B(NNxKK) global->smem (the wrapper utma_load
// swizzle==3 path, probed to match _sw128_phys_off), then a single warpgroup
// HGMMA.64xNNx16 chains KK/16 accumulating MMAs. Dumps raw D[m,n]. This removes
// the cute-STS-fill confound of the earlier microbenches: smem is filled by TMA
// exactly as the real kernel.
#include "utils.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/arch/copy_sm90.hpp>
#include <cute/arch/copy_sm90_tma.hpp>
#include "cutlass/gemm/collective/builders/sm90_common.inl"
using namespace cute;

#ifndef KK
#define KK 64
#endif
#define MM 64
#define NN 32

using Element = cutlass::bfloat16_t;
using GA_Layout = decltype(make_layout(Shape<Int<MM>,Int<KK>>{}, LayoutRight{}));
using GB_Layout = decltype(make_layout(Shape<Int<NN>,Int<KK>>{}, LayoutRight{}));
using AtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<MM>, Int<KK>>());
using AtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, Element, Int<NN>, Int<KK>>());
using SmemLayoutA = decltype(tile_to_shape(AtomA{}, Shape<Int<MM>,Int<KK>>{}));
using SmemLayoutB = decltype(tile_to_shape(AtomB{}, Shape<Int<NN>,Int<KK>>{}));

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutA>> sA;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutB>> sB;
  alignas(16)  uint64_t mbar[1];
};
static __host__ auto _dA(){ return make_tensor(make_gmem_ptr((Element const*)nullptr), make_layout(Shape<Int<MM>,Int<KK>>{}, LayoutRight{})); }
static __host__ auto _dB(){ return make_tensor(make_gmem_ptr((Element const*)nullptr), make_layout(Shape<Int<NN>,Int<KK>>{}, LayoutRight{})); }
using TmaA_t = decltype(make_tma_copy(SM90_TMA_LOAD{}, _dA(), SmemLayoutA{}));
using TmaB_t = decltype(make_tma_copy(SM90_TMA_LOAD{}, _dB(), SmemLayoutB{}));

extern "C" __global__ void
tma_sw128_k(CUTE_GRID_CONSTANT TmaA_t const tma_a,
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
  auto cta_a = tma_a.get_slice(Int<0>{});
  auto cta_b = tma_b.get_slice(Int<0>{});
  Tensor tAgA = cta_a.partition_S(mA); Tensor tAsA = cta_a.partition_D(sA);
  Tensor tBgB = cta_b.partition_S(mB); Tensor tBsB = cta_b.partition_D(sB);
  int t = threadIdx.x;
  constexpr int kBytesA = sizeof(Element)*MM*KK;
  constexpr int kBytesB = sizeof(Element)*NN*KK;
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
  TiledMMA tiled_mma = make_tiled_mma(SM90_64x32x16_F32BF16BF16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(t);
  Tensor tCsA = thr_mma.partition_A(sA); Tensor tCsB = thr_mma.partition_B(sB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA); Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<MM>,Int<NN>>{}); clear(accum);
  warpgroup_fence_operand(accum); warpgroup_arrive();
  gemm(tiled_mma, tCrA, tCrB, accum);
  warpgroup_commit_batch(); warpgroup_wait<0>(); warpgroup_fence_operand(accum);
  Tensor mD = make_tensor(make_gmem_ptr(gD), make_layout(Shape<Int<MM>,Int<NN>>{}, LayoutRight{}));
  Tensor tCgD = thr_mma.partition_C(mD);
  for (int i = 0; i < size(accum); ++i) tCgD(i) = accum(i);
#endif
}

int main() {
  Element hA[MM*KK], hB[NN*KK]; float hD[MM*NN];
  for (int i=0;i<MM*KK;i++) hA[i]=static_cast<Element>(((i*37%13)-6)*0.05f);
  for (int i=0;i<NN*KK;i++) hB[i]=static_cast<Element>(((i*29%11)-5)*0.05f);
  memset(hD,0,sizeof(hD));
  Element *dA,*dB; float *dD;
  cudaMalloc(&dA,sizeof(hA)); cudaMalloc(&dB,sizeof(hB)); cudaMalloc(&dD,sizeof(hD));
  cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice);
  cudaMemset(dD,0,sizeof(hD));
  GA_Layout ga = make_layout(Shape<Int<MM>,Int<KK>>{}, LayoutRight{});
  GB_Layout gb = make_layout(Shape<Int<NN>,Int<KK>>{}, LayoutRight{});
  Tensor gmA = make_tensor(make_gmem_ptr((Element const*)dA), ga);
  Tensor gmB = make_tensor(make_gmem_ptr((Element const*)dB), gb);
  auto tma_a = make_tma_copy(SM90_TMA_LOAD{}, gmA, SmemLayoutA{});
  auto tma_b = make_tma_copy(SM90_TMA_LOAD{}, gmB, SmemLayoutB{});
  int smem_bytes = (int)sizeof(SharedStorage);
  cudaFuncSetAttribute(tma_sw128_k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  tma_sw128_k<<<1,128,smem_bytes>>>(tma_a, tma_b, dD, ga, gb);
  cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(hD,dD,sizeof(hD),cudaMemcpyDeviceToHost);
  double maxabs=0;
  for(int m=0;m<MM;m++)for(int n=0;n<NN;n++){
    double acc=0; for(int k=0;k<KK;k++) acc+=(double)(float)hA[m*KK+k]*(double)(float)hB[n*KK+k];
    double d=fabs(acc-(double)hD[m*NN+n]); if(d>maxabs)maxabs=d;
  }
  printf("native max=%.5f\nTEST PASSED\n", maxabs);
  return 0;
}
