/*
 * ML Tensor Core batched GEMM micro-kernel (FP16 x FP16 -> FP32).
 */

#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace {
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;
constexpr int kBatchCount = 3;
constexpr int kTileElems = kM * kN;
}  // namespace

extern "C" __global__ void ml_tensorop_batched_gemm_f16(const half* a,
                                                              const half* b,
                                                              float* c) {
  if (blockIdx.x >= kBatchCount || threadIdx.x >= 32) {
    return;
  }

  const half* a_batch = a + blockIdx.x * kTileElems;
  const half* b_batch = b + blockIdx.x * kTileElems;
  float* c_batch = c + blockIdx.x * kTileElems;

  wmma::fragment<wmma::matrix_a, kM, kN, kK, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, kM, kN, kK, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, kM, kN, kK, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);
  wmma::load_matrix_sync(a_frag, a_batch, kK);
  wmma::load_matrix_sync(b_frag, b_batch, kK);
  wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  wmma::store_matrix_sync(c_batch, c_frag, kN, wmma::mem_row_major);
}
