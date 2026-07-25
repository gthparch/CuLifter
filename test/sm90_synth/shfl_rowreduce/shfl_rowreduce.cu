// shfl_rowreduce — isolates the 41_fmha softmax QUAD butterfly reduction.
//
// 41_fmha's softmax rowsum reduces a 4-lane QUAD via SHFL.BFLY ^1 then ^2
// (width clamp 0x1f => full-warp segment, but only 2 stages => sums lanes
// {4q, 4q+1, 4q+2, 4q+3}). Each quad q holds one "row"'s 4 column-partials.
//
// To expose any quad-specific / row-position lane-mapping error, each lane's
// input is ROW-POSITION-VARYING: in[tid] is distinct per lane, so the per-quad
// sum is a position-dependent value. A lane-mapping bug => wrong sum for the
// quads at specific positions (mirroring 41_fmha rows {34,40}). Every lane of
// a quad writes the reduced quad-sum to out[tid]; oracle checks all of them.
extern "C" __global__ void shfl_rowreduce(const float* in, float* out, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? in[tid] : 0.0f;
    // EXACT 41_fmha quad butterfly: xor 1 then xor 2 (full-warp clamp).
    val += __shfl_xor_sync(0xffffffff, val, 1);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    if (tid < N) out[tid] = val;
}
