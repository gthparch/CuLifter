// Microbench: divide by warpSize (a RUNTIME builtin in CUDA — ptxas emits
// the magic-rcp chain, and on sm_90 hoists the magic into UNIFORM registers:
// F2I -> R2UR -> UIMAD.WIDE.U32 Newton refinement; the uniform quotient
// (blockDim/warpSize) is the HI WORD of a UIMAD.WIDE.U32 pair). This is the
// cudnn so.69 bn_bw_1C11 warpid chain shape IntDivRecognize must match.
extern "C" __global__ void kern_warpdiv(int* out_warpid, int* out_nwarps,
                                        int* out_lane, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = threadIdx.x / warpSize;   // vector dividend, magic in UR
    int nwarps = blockDim.x / warpSize;    // uniform dividend, UIMAD.WIDE leaf
    int lane   = threadIdx.x % warpSize;
    if (tid < N) {
        out_warpid[tid] = warpid;
        out_nwarps[tid] = nwarps;
        out_lane[tid]   = lane;
    }
}
