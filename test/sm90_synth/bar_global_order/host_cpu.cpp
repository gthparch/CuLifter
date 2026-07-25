// host_cpu.cpp — driver for bar_global_order.
//
// Reproduces the GENERIC warp-lockstep global-store ordering bug (cuDNN
// compute_alphas_and_betas<float> alphas[1] race): TWO distinct static global
// stores, executed by TWO lanes of ONE warp, write the SAME global cell out[1]
// with NO block barrier between them. On a real GPU the warp runs in SIMT
// lockstep, so the LATER-PC store wins deterministically (out[1]==901.0 every
// run). The x86 wrapper runs CTA threads as concurrent std::threads; lanes
// desync and the two stores interleave -> last-writer race.
//
//   out[1] earlier-PC writer: lane s=1 -> out[1] = 101.0 (INIT)
//   out[1] later-PC  writer: lane s=0 -> out[1] = 901.0 (RECUR)  <-- GPU wins
//
// FIX: the lockstep turnstile, extended to bracket GLOBAL stores
// (lift with CULIFTER_LOCKSTEP=1, run with SLIFTER_LOCKSTEP=1). Under it the two
// stores retire in warp round-order and out[1]==901.0 deterministically.
//
// Pass criterion:
//   - lockstep ON  (SLIFTER_LOCKSTEP set): REQUIRE out[1]==901.0 every run (fix works).
//   - lockstep OFF (default): the race is the documented un-fixed default; we
//     REPORT the distribution and PASS (the fix is opt-in; nothing here regresses).
//
// sm_90 layout: args at c[0x0][0x210]; pinned via ArgAt.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void bar_global_order();

int main(int argc, char** argv) {
    constexpr int S = 8;            // blockDim.x (8 threads = 1 warp, like CTC BLK=8)
    constexpr int T = 3;            // unused by the kernel body, kept for the arg layout
    const float INIT_VAL  = 101.0f; // earlier-PC writer (lane s=1) of out[1]
    const float RECUR_VAL = 901.0f; // later-PC  writer (lane s=0) of out[1] — GPU wins

    int runs = 256;
    if (argc > 1) runs = std::atoi(argv[1]);
    if (const char* e = std::getenv("SLIFTER_RUNS")) runs = std::atoi(e);
    const bool lockstep = std::getenv("SLIFTER_LOCKSTEP") != nullptr;

    float* out = new float[8];
    int n_recur = 0, n_init = 0, n_other = 0;
    float first = 0.0f; bool deterministic = true;

    for (int r = 0; r < runs; r++) {
        for (int i = 0; i < 8; i++) out[i] = -7.0f;   // canary
        launchKernel(bar_global_order, 1, S,
                     ArgAt{0x210}, out, ArgAt{0x218}, (int)S, ArgAt{0x21c}, (int)T);
        float v = out[1];
        if (r == 0) first = v; else if (v != first) deterministic = false;
        if      (std::fabs(v - RECUR_VAL) < 1e-3f) n_recur++;
        else if (std::fabs(v - INIT_VAL)  < 1e-3f) n_init++;
        else                                       n_other++;
    }

    printf("lockstep=%s runs=%d  out[1]: recur(901)=%d init(101)=%d other=%d  deterministic=%s\n",
           lockstep ? "on" : "off", runs, n_recur, n_init, n_other, deterministic ? "yes" : "NO");

    bool ok;
    if (lockstep) {
        // Fix must hold: every run produces the GPU-correct later-PC (recur) value.
        ok = (n_recur == runs) && deterministic;
        printf("%s\n", ok ? "TEST PASSED"
            : "DIVERGED (lockstep global-store bracket did not order the warp-synchronous stores)");
    } else {
        // Default (un-fixed, opt-in) path: the race is expected; report + pass.
        ok = (n_other == 0);   // sanity: every run lands on one of the two known values
        printf("RACE REPRODUCED (default path; fix is opt-in via CULIFTER_LOCKSTEP+SLIFTER_LOCKSTEP)\n");
        printf("%s\n", ok ? "TEST PASSED" : "DIVERGED (unexpected out[1] value)");
    }
    delete[] out;
    return ok ? 0 : 1;
}
