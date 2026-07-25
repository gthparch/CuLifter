// Multi-warp single-mma test: 16x32 output, 2 warps, single mma_sync per warp.
// Used to isolate multi-warp bugs vs single-warp + single-mma_sync (= tensor_simple).
#include <mma.h>
using namespace nvcuda;

extern "C" __global__ void wmma_2warp(half *a, half *b, float *c) {
    int warpId = (threadIdx.x) / 32;
    // 2 warps -> 2 N-tile columns of 16x16 each = (16x16) and (16x16) covering 16x32 C
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0f);
    // A is 16x16, shared by both warps.
    wmma::load_matrix_sync(af, a, 16);
    // B is 16x32 stride 32; warp 0 reads cols 0..15, warp 1 reads cols 16..31.
    wmma::load_matrix_sync(bf, b + warpId * 16, 32);
    wmma::mma_sync(cf, af, bf, cf);
    // C is 16x32 stride 32; warp 0 writes cols 0..15, warp 1 writes cols 16..31.
    wmma::store_matrix_sync(c + warpId * 16, cf, 32, wmma::mem_row_major);
}
