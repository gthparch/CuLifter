// bar_global_order — minimal reproduction of the GENERIC warp-lockstep
// global-store ordering bug (the global-store analog of the warp-membership
// mass-drop that warp_gather_fix fixed).
//
// Root cause (localized from cuDNN compute_alphas_and_betas<float> alphas[1]):
//   TWO DIFFERENT static global stores, executed by TWO lanes of ONE warp, write
//   the SAME global cell with NO block barrier between them. On a real GPU the
//   warp runs in SIMT LOCKSTEP, so the two stores retire in PC/program order
//   within the warp and the LATER store wins deterministically. The x86 wrapper
//   runs each lane as an independent std::thread; the lanes desynchronize and the
//   two stores interleave in arbitrary order -> last-writer race.
//
// CTC alias reproduced (sid17 init by tid1 vs sid22 recursion by tid0):
//   lane s=1 takes the INIT  branch -> writes out[1] = 101.0   (earlier in PC)
//   lane s=0 takes the RECUR branch -> writes out[1] = 901.0   (later in PC, GPU wins)
// The two stores are in DIFFERENT branches (different PCs), NOT separated by a
// barrier — exactly like CTC's two aliasing alpha writes inside the DP region.
//
// GPU (single-warp lockstep): out[1] == 901.0 (later-PC store) EVERY run.
// Buggy wrapper (default): out[1] non-deterministic (the INIT 101.0 can win).
// Fix: bracket global stores with the lockstep turnstile (CULIFTER_LOCKSTEP=1 +
// SLIFTER_LOCKSTEP=1) so they retire in warp round-order.
//
// Block = 8 threads (1 warp, like CTC BLK=8).
extern "C" __global__ void bar_global_order(float* __restrict__ out, int S, int T) {
    int s = threadIdx.x;
    // A little shared traffic up front (as in CTC), then the two unsynchronized
    // global stores. NO __syncthreads() between the two stores below.
    __shared__ float sh[8];
    if (s < S) sh[s] = (float)s;
    __syncthreads();

    // TWO distinct static global stores aliasing out[1], in different branches
    // (different PCs), with NO barrier between them. Warp lockstep is the only
    // thing that orders them on GPU.
    if (s == 1) {
        // INIT writer (earlier branch / lower PC): lane 1 -> out[1] = 101.0
        out[1] = 100.0f + (float)s;            // = 101.0
    } else if (s == 0) {
        // do a few dependent ops so this lane's store is at a LATER PC and takes
        // longer (amplifies the desync window in the wrapper).
        float acc = sh[0];
        #pragma unroll
        for (int k = 0; k < 8; k++) acc = acc * 1.0009f + 0.0001f;
        // RECUR writer (later branch / higher PC): lane 0 -> out[1] = 901.0
        out[1] = 900.0f + 1.0f + (acc - acc); // = 901.0 (acc-acc keeps acc live)
    }
    // out[0], out[2..] left as the host canary; only out[1] is the contended cell.
}
