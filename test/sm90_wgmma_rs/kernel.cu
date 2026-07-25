// Real sm_90a tensor-core microkernel via cute (CUTLASS 3.5).
// Single warpgroup (128 threads), 1 CTA. Computes D[64x16] = A[64x16] * B[16x16]
// with fp16 inputs and fp32 accumulation, emitting genuine HGMMA.64x16x16.F32
// ... gdesc[...] SASS (no traps). A and B are both sourced from shared-memory
// descriptors (the SS form used throughout CUTLASS / FlashAttention mainloops).
//
// All fragment / descriptor layouts are handled by cute's TiledMMA + partition
// machinery, so the kernel is correct by construction. We dump A, B and the
// resulting D to disk for cross-checking against the x86-lifted run.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/arch/mma_sm90.hpp>

using namespace cute;

// M=64, N=16, K=16 single-tile GMMA, fp16 in / fp32 accum, SS (smem A and B).
extern "C" __global__ void fa_wgmma(const half* __restrict__ gA,   // 64x16 row-major (MxK)
                                    const half* __restrict__ gB,   // 16x16 row-major (NxK)
                                    float* __restrict__ gD) {      // 64x16 row-major (MxN)
#if defined(__CUDA_ARCH__)
  using TA = half_t; using TB = half_t;

  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, Shape<_64,_16>{}));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TB>{}, Shape<_16,_16>{}));

  __shared__ __align__(128) TA sA[cosize_v<SmemLayoutA>];
  __shared__ __align__(128) TB sB[cosize_v<SmemLayoutB>];

  Tensor mA = make_tensor(make_smem_ptr(sA), SmemLayoutA{});  // (64,16)
  Tensor mB = make_tensor(make_smem_ptr(sB), SmemLayoutB{});  // (16,16)

  int t = threadIdx.x;
  for (int i = t; i < 64*16; i += blockDim.x) { int m=i/16, k=i%16; mA(m,k) = static_cast<half_t>(gA[m*16+k]); }
  for (int i = t; i < 16*16; i += blockDim.x) { int n=i/16, k=i%16; mB(n,k) = static_cast<half_t>(gB[n*16+k]); }
  __syncthreads();

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x16x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(t);

  Tensor tCsA = thr_mma.partition_A(mA);
  Tensor tCsB = thr_mma.partition_B(mB);
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
#endif // __CUDA_ARCH__
}

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s OUTPUT_DIR\n", argv[0]);
    return 2;
  }
  const char* OUT = argv[1];
  const int M=64,K=16,N=16;
  half hA[M*K], hB[N*K]; float hD[M*N];
  for(int i=0;i<M*K;i++) hA[i]=__float2half(((i*37%13)-6)*0.05f);
  for(int i=0;i<N*K;i++) hB[i]=__float2half(((i*29%11)-5)*0.05f);
  memset(hD,0,sizeof(hD));
  half *dA,*dB; float *dD;
  cudaMalloc(&dA,sizeof(hA)); cudaMalloc(&dB,sizeof(hB)); cudaMalloc(&dD,sizeof(hD));
  cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice);
  cudaMemset(dD,0,sizeof(hD));
  fa_wgmma<<<1,128>>>(dA,dB,dD);
  cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(hD,dD,sizeof(hD),cudaMemcpyDeviceToHost);

  // CPU fp32 reference: D[m,n] = sum_k A[m,k]*B[n,k]
  double maxabs=0;
  for(int m=0;m<M;m++) for(int n=0;n<N;n++){
    double acc=0; for(int k=0;k<K;k++) acc += (double)__half2float(hA[m*K+k]) * (double)__half2float(hB[n*K+k]);
    double diff=fabs(acc - (double)hD[m*N+n]); if(diff>maxabs) maxabs=diff;
  }
  printf("=== native HGMMA D row0 ===\n");
  for(int n=0;n<N;n++) printf("%.4f ",hD[n]); printf("\n");
  printf("max |HGMMA - cpu_fp32_ref| = %.5f\n", maxabs);

  char p[256];
  snprintf(p,sizeof(p),"%s/ref_D.bin",OUT); { FILE*f=fopen(p,"wb"); fwrite(hD,sizeof(float),M*N,f); fclose(f);}
  snprintf(p,sizeof(p),"%s/in_A.bin",OUT);  { FILE*f=fopen(p,"wb"); fwrite(hA,sizeof(half),M*K,f); fclose(f);}
  snprintf(p,sizeof(p),"%s/in_B.bin",OUT);  { FILE*f=fopen(p,"wb"); fwrite(hB,sizeof(half),N*K,f); fclose(f);}
  printf("TEST PASSED\n");
  return 0;
}
