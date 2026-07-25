// cross_cta_spin.cu — MINIMAL reproduction of the stream-K cross-CTA spin-barrier
// deadlock on the x86 wrapper (the 47_streamk motivating case).
//
// Stream-K's partial-tile fixup: an OWNER CTA collects partial-K accumulators
// that PEER CTAs deposit. The peers atomic-add (REDG.E.ADD) their partial into a
// global workspace, then set a barrier flag (ATOMG.CAS / atomic store) under a GPU
// release fence; the owner SPINS on an acquire-load of that flag until all peers
// have signalled, then reduces + writes the result.  This is a TRUE in-kernel
// cross-CTA reduction + spin-barrier + GPU fence.
//
// On a real GPU all CTAs are co-resident (or at least make forward progress
// concurrently), so the owner's spin resolves once the peer signals.  The x86
// wrapper runs CTAs SEQUENTIALLY — CTA c runs to completion before CTA c+1 starts.
// So if the OWNER (low CTA id) is scheduled first and waits on a flag set by a
// PEER (high CTA id) that has not run yet, the spin is a REAL infinite loop.
//
// This kernel is the minimal shape: 2 CTAs, 1 thread each.
//   CTA 1 (PRODUCER): atomicAdd(&partials[0], my_partial); fence; flag store/CAS.
//   CTA 0 (OWNER):    spin on acquire-load of flag; fence; out[0] = partials[0]+base.
// Under the ascending sequential CTA loop CTA 0 runs first → spins forever on a
// flag CTA 1 will only set later → DEADLOCK (wrapper hangs; the test times out).
//
// A `mode` arg lets the SAME kernel run a deadlock-free SINGLE-CTA control:
//   mode==0 : cross-CTA spin (the deadlock repro; needs concurrent CTAs)
//   mode==1 : single-CTA self-contained (owner==producer, no peer wait) — the
//             control that proves the kernel/atomics/lift are otherwise correct
//             and exercises REDG/ATOMG/LDG-acquire on the SAME (sequential) path.
//
// args (sm_90 layout, pinned via ArgAt in host_cpu.cpp):
//   out       : float*  result (owner writes out[0])
//   partials  : float*  cross-CTA partial-accumulator workspace (1 cell)
//   flag      : int*    barrier flag workspace (1 cell, 0 = unset)
//   base      : float   owner's own contribution (added to the collected partial)
//   contrib   : float   producer's partial contribution
//   mode      : int     0 = cross-CTA spin, 1 = single-CTA control

extern "C" __global__ void cross_cta_spin(float* out, float* partials, int* flag,
                                          float base, float contrib, int mode) {
    int cta = blockIdx.x;

    if (mode == 1) {
        // SINGLE-CTA control: one CTA does the whole thing, no peer wait.
        // Still exercises the atomic-add partial + acquire-load + finalize on the
        // sequential path (no deadlock — the producer and owner are the same CTA).
        atomicAdd(&partials[0], contrib);       // REDG-style partial deposit
        __threadfence();                         // MEMBAR.ALL.GPU release
        atomicExch(&flag[0], 1);                 // set barrier flag
        // acquire-load spin (resolves immediately — same CTA already set it)
        while (atomicAdd(&flag[0], 0) == 0) { }  // ATOMG-style acquire poll
        __threadfence();                         // MEMBAR.ALL.GPU acquire
        out[0] = partials[0] + base;             // finalize
        return;
    }

    // mode==0: TRUE cross-CTA spin (the deadlock repro).
    if (cta == 1) {
        // PRODUCER: deposit partial, fence, signal.
        atomicAdd(&partials[0], contrib);        // REDG.E.ADD into partials_workspace
        __threadfence();                          // MEMBAR.ALL.GPU (release)
        atomicExch(&flag[0], 1);                  // ATOMG: set the barrier flag
    } else {
        // OWNER (cta==0): SPIN on the flag until the peer (cta==1) signals.
        // Under the sequential wrapper cta==0 runs FIRST → flag never set → hang.
        while (atomicAdd(&flag[0], 0) == 0) { }   // acquire-load spin (LDG.E.STRONG)
        __threadfence();                          // MEMBAR.ALL.GPU (acquire)
        out[0] = partials[0] + base;              // reduce + write D
    }
}
