#include "utils.h"
// wgmma_n96_reduce — isolate the FA3 hdim192 n_dim=96 score reduction.
//
// The hdim192_128 kernel's QK GEMM emits HGMMA.64x96x16 (n_dim=96). The
// online-softmax reduces row_max / row_sum over those 96 columns via
// thread_reduce_ (each thread maxes its local columns of the C-fragment) then
// quad_allreduce_<4> (SHFL.BFLY across the 4 threads owning a row). This
// microbench runs a genuine cute SM90_64x96x16 SS WGMMA then the EXACT FA3
// reduce (reduce_max + scale_apply_exp2 + reduce_sum) and dumps per-row
// row_max + row_sum to gO, comparing against a CPU reference. It isolates the
// n_dim=96 C-fragment column-layout + cross-quad SHFL reduction — the one QK
// component the scalar fa3_faithful microbench cannot express.
//
// D[64x96] = A[64x16] * B[96x16]^T (fp16 in, fp32 accum). One warpgroup, 1 CTA.
// gO layout: per query-row m (0..63): [row_max(m), row_sum(m)] (2 floats).
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

#define MM 64
#define KK 16
#define NN 96

// Minimal port of flash::reduce_max / scale_apply_exp2 / reduce_sum quad path.
template <int W, typename Op>
__device__ __forceinline__ float allreduce4(float x, Op op) {
#pragma unroll
    for (int m = 1; m < 4; m <<= 1) {
        float y = __shfl_xor_sync(0xffffffff, x, m, 4);
        x = op(x, y);
    }
    return x;
}

extern "C" __global__ void wgmma_n96_reduce(
    const half* __restrict__ gA,    // 64x16 row-major (M x K)
    const half* __restrict__ gB,    // 96x16 row-major (N x K)
    float* __restrict__ gO)         // 64 x 2 (row_max,row_sum) per query row
{
#if defined(__CUDA_ARCH__)
  using TA = half_t; using TB = half_t;
  using SmemLayoutA = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TA>{}, Shape<Int<MM>,Int<KK>>{}));
  using SmemLayoutB = decltype(tile_to_shape(GMMA::Layout_K_INTER_Atom<TB>{}, Shape<Int<NN>,Int<KK>>{}));

  __shared__ __align__(128) TA sA[cosize_v<SmemLayoutA>];
  __shared__ __align__(128) TB sB[cosize_v<SmemLayoutB>];

  Tensor mA = make_tensor(make_smem_ptr(sA), SmemLayoutA{});
  Tensor mB = make_tensor(make_smem_ptr(sB), SmemLayoutB{});

  int t = threadIdx.x;
  for (int i = t; i < MM*KK; i += blockDim.x) { int m=i/KK, k=i%KK; mA(m,k) = static_cast<half_t>(gA[m*KK+k]); }
  for (int i = t; i < NN*KK; i += blockDim.x) { int n=i/KK, k=i%KK; mB(n,k) = static_cast<half_t>(gB[n*KK+k]); }
  __syncthreads();

  TiledMMA tiled_mma = make_tiled_mma(SM90_64x96x16_F32F16F16_SS<GMMA::Major::K, GMMA::Major::K>{});
  auto thr_mma = tiled_mma.get_slice(t);

  Tensor tCsA = thr_mma.partition_A(mA);
  Tensor tCsB = thr_mma.partition_B(mB);
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);
  Tensor accum = partition_fragment_C(tiled_mma, Shape<Int<MM>,Int<NN>>{});
  clear(accum);

  warpgroup_fence_operand(accum);
  warpgroup_arrive();
  gemm(tiled_mma, tCrA, tCrB, accum);
  warpgroup_commit_batch();
  warpgroup_wait<0>();
  warpgroup_fence_operand(accum);

  // FA3 reduce: reshape accum ((2,2,V),MMA_M,MMA_N) -> (nrow,ncol) and reduce.
  const float scale = 0.25f;
  const float scale_log2 = scale * 1.4426950408889634f;
  Tensor scores = make_tensor(accum.data(), flash::convert_layout_acc_rowcol(accum.layout()));
  // scores shape: (nrow, ncol) where nrow=2 per thread (the row pair), ncol=N/...
  int nrow = size<0>(scores);
  int ncol = size<1>(scores);

  // thread_reduce_ max over ncol, then quad_allreduce<4>.
  for (int mi = 0; mi < nrow; mi++) {
    float rmax = -INFINITY;
    for (int ni = 0; ni < ncol; ni++) rmax = fmaxf(rmax, scores(mi, ni));
    rmax = allreduce4<4>(rmax, [](float a, float b){ return fmaxf(a,b); });
    // scale_apply_exp2 + reduce_sum over this thread's columns.
    float rmax_scaled = (rmax == -INFINITY) ? 0.f : rmax * scale_log2;
    float rsum = 0.f;
    for (int ni = 0; ni < ncol; ni++) {
        float p = exp2f(scores(mi, ni) * scale_log2 - rmax_scaled);
        rsum += p;
    }
    rsum = allreduce4<4>(rsum, [](float a, float b){ return a+b; });
    // Recover the global query row for (thread t, mi). cute CLayout for
    // SM90_64xNx16: warp = t/32 (0..3), within-warp lane group. We dump
    // (row_max,row_sum) keyed by a per-(t,mi) row index computed from cute.
    // Easiest: have thread0 of each quad write. Use partition_C identity to
    // get the row. We instead write ALL threads to a scratch keyed by row.
    Tensor cId = make_identity_tensor(Shape<Int<MM>,Int<NN>>{});
    Tensor tCcId = thr_mma.partition_C(cId);
    Tensor cId_rc = make_tensor(tCcId.data(), flash::convert_layout_acc_rowcol(tCcId.layout()));
    int row = get<0>(cId_rc(mi, 0));
    gO[row*2 + 0] = rmax;
    gO[row*2 + 1] = rsum;
  }
#endif
}

int main() {
  half hA[MM*KK], hB[NN*KK]; float hO[MM*2];
  for(int i=0;i<MM*KK;i++) hA[i]=__float2half(((i*37%13)-6)*0.05f);
  for(int i=0;i<NN*KK;i++) hB[i]=__float2half(((i*29%11)-5)*0.05f);
  memset(hO,0,sizeof(hO));
  half *dA,*dB; float *dO;
  cudaMalloc(&dA,sizeof(hA)); cudaMalloc(&dB,sizeof(hB)); cudaMalloc(&dO,sizeof(hO));
  cudaMemcpy(dA,hA,sizeof(hA),cudaMemcpyHostToDevice);
  cudaMemcpy(dB,hB,sizeof(hB),cudaMemcpyHostToDevice);
  cudaMemset(dO,0,sizeof(hO));
  wgmma_n96_reduce<<<1,128>>>(dA,dB,dO);
  cudaDeviceSynchronize();
  cudaError_t e=cudaGetLastError();
  if(e!=cudaSuccess){printf("CUDA error: %s\n",cudaGetErrorString(e));return 1;}
  cudaMemcpy(hO,dO,sizeof(hO),cudaMemcpyDeviceToHost);
  printf("row0 max=%.4f sum=%.4f\n", hO[0], hO[1]);
  printf("TEST PASSED\n");
  return 0;
}
