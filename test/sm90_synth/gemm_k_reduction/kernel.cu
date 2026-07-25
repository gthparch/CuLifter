// CUTLASS 23_ampere_gemm_operand_reduction_fusion K-REDUCTION audit.
//
// Reproduces the operand-A K-reduction's distinct surface:
//   (1) warp-level partial: each lane sums its bf16 A-fragment elements to f32
//       (mma_with_reduction_tensor_op.h: gemm_k_reduction[m] += float(A[..])),
//   (2) epilogue cross-lane butterfly reduction via __shfl_xor_sync (xor 1, then
//       xor 2) over each group of 4 lanes (epilogue_gemm_k_reduction.h:158-159),
//   (3) col_ = lane&3 selects which butterfly result is the row's reduction,
//   (4) convert f32 -> bf16 and store (the per-row reduction vector).
//
// The distinct ops audited: SHFL.BFLY xor-butterfly warp reduction + bf16
// fragment decode + bf16 output convert. (The HMMA accumulate is audited
// elsewhere; here we isolate the reduction path.)
//
// Layout: one warp (32 lanes). Each lane owns 8 bf16 A-fragment elements
// Afrag[lane*8 + 0..7]. For row-group g = lane>>2, the 4 lanes (col 0..3) each
// hold partial sums; the xor-butterfly sums all 4 -> the full row reduction,
// which col_==c picks for output row (g*4 + c is the logical reduction index).
//
// out[g*4 + col_] = bf16( sum_over_the_4_lanes_in_group_g( lane_partial ) )
//   where lane_partial = sum_{e in {0,1,4,5}} A[lane*8+e]  (the m*2 even-row sum)
// (we audit the even-row m*2 accumulation pattern: A[8m], A[8m+1], A[8m+4], A[8m+5].)
#include <cuda_bf16.h>

extern "C" __global__ void kern_k_reduction(
    const __nv_bfloat16* __restrict__ Afrag,   // [32*8] per-lane A fragments
    __nv_bfloat16* __restrict__ out,            // [32] per-(group,col) reduction
    int dummy)
{
    const int lane = threadIdx.x & 31;
    const int g    = lane >> 2;     // row group (0..7)
    const int col_ = lane & 3;      // column within quad (0..3)

    // (1) warp-level partial: this lane's bf16 fragment sum -> f32
    //     (mma_with_reduction even-row pattern: A[0],A[1],A[4],A[5])
    const __nv_bfloat16* A = Afrag + lane * 8;
    float partial = 0.f;
    partial += __bfloat162float(A[0]);
    partial += __bfloat162float(A[1]);
    partial += __bfloat162float(A[4]);
    partial += __bfloat162float(A[5]);

    // (2) epilogue butterfly: reduce across the 4 lanes of the quad
    float sum = partial;
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);

    // (3)(4) every lane in the quad now holds the full quad-sum; write per lane
    //         (col_ selects the logical row; here all 4 lanes hold the same sum
    //          so out[g*4+col_] is the group's reduction at each col slot).
    out[g * 4 + col_] = __float2bfloat16(sum);
}
