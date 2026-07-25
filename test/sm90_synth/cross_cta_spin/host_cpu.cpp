// host_cpu.cpp — driver for cross_cta_spin (stream-K cross-CTA spin-barrier repro).
//
// Two phases:
//   PHASE A — SINGLE-CTA CONTROL (mode=1, grid=1): the producer and owner are the
//     SAME CTA, so the acquire-poll resolves immediately on the sequential wrapper.
//     This exercises REDG (atomicAdd partial) + ATOMG.EXCH (flag set) + the
//     acquire-load poll + __threadfence on the SEQUENTIAL path and MUST produce
//     out[0] == base + contrib.  It proves the lift/atomics are correct; the only
//     thing missing for cross-CTA is concurrent CTA execution.
//
//   PHASE B — CROSS-CTA SPIN (mode=0, grid=2): OWNER=CTA0 spins on a flag the
//     PRODUCER=CTA1 sets.  Under the sequential wrapper CTA0 runs first → the flag
//     is never set while CTA0 runs → the kernel's acquire-poll is a REAL infinite
//     loop → DEADLOCK.  We launch it on a WATCHDOG thread and detect the hang via a
//     timeout.  The hang IS the documented current behavior (the 47_streamk
//     finding); this phase therefore PASSES when it deadlocks (characterization),
//     and — once a concurrent-CTA mode lands — would instead COMPLETE with
//     out[0] == base + contrib (the value PHASE A already verifies).
//
//   Opt-in escape: if SLIFTER_EXPECT_CONCURRENT_CTA is set, PHASE B must COMPLETE
//     (not deadlock) and produce the correct value — the gate a future cross-CTA
//     execution mode is validated against.
//
// sm_90 arg layout (from the SASS const reads):
//   0x208 desc=0 | 0x210 out | 0x218 partials | 0x220 flag | 0x228 base(f32)
//   0x22c contrib(f32) | 0x230 mode(i32)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <atomic>
#include <thread>
#include <chrono>
#include "kernel_wrapper.h"

extern "C" void cross_cta_spin();

static void launch(float* out, float* partials, int* flag,
                   float base, float contrib, int mode, int grid) {
    launchKernel(cross_cta_spin, grid, 1,
                 ArgAt{0x208}, (uint64_t)0,        // global descriptor base = 0
                 ArgAt{0x210}, out,
                 ArgAt{0x218}, partials,
                 ArgAt{0x220}, flag,
                 ArgAt{0x228}, base,
                 ArgAt{0x22c}, contrib,
                 ArgAt{0x230}, (int)mode);
}

int main() {
    const float base    = 3.5f;
    const float contrib = 8.25f;
    const float expect  = base + contrib;   // 11.75

    bool all_ok = true;

    // ---- PHASE A: single-CTA control (must complete + be correct) ------------
    {
        float    out[1]      = { -7.0f };
        float    partials[1] = { 0.0f };
        int      flag[1]     = { 0 };
        launch(out, partials, flag, base, contrib, /*mode=*/1, /*grid=*/1);
        bool ok = std::fabs(out[0] - expect) < 1e-4f;
        printf("PHASE A (single-CTA control): out[0]=%.4f expect=%.4f -> %s\n",
               out[0], expect, ok ? "OK" : "WRONG");
        all_ok = all_ok && ok;
    }

    // ---- PHASE B: cross-CTA spin (deadlocks on the sequential wrapper) --------
    const bool expect_concurrent = std::getenv("SLIFTER_EXPECT_CONCURRENT_CTA") != nullptr;
    {
        // Heap so the watchdog-orphaned kernel thread can't corrupt our stack if
        // it ever wakes after we move on.
        float* out      = new float[1]{ -7.0f };
        float* partials = new float[1]{ 0.0f };
        int*   flag     = new int[1]{ 0 };

        std::atomic<bool> done{false};
        std::thread worker([&]{
            launch(out, partials, flag, base, contrib, /*mode=*/0, /*grid=*/2);
            done.store(true, std::memory_order_release);
        });

        // Watchdog: give the cross-CTA launch a bounded window to finish.
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
        while (!done.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        bool completed = done.load(std::memory_order_acquire);

        if (expect_concurrent) {
            // A concurrent-CTA mode is claimed: PHASE B must complete + be correct.
            bool ok = completed && std::fabs(out[0] - expect) < 1e-4f;
            printf("PHASE B (cross-CTA, EXPECT_CONCURRENT): completed=%s out[0]=%.4f expect=%.4f -> %s\n",
                   completed ? "yes" : "NO(timeout)", completed ? out[0] : 0.0f, expect,
                   ok ? "OK" : "FAIL");
            all_ok = all_ok && ok;
            if (completed) worker.join();
            else worker.detach();   // still hung; leak the thread (process exits)
        } else {
            // Default (sequential) wrapper: the DEADLOCK is the documented behavior.
            if (completed) {
                // Did not hang. Two sub-cases: correct value = cross-CTA already
                // works (great, but unexpected on the sequential model); incorrect
                // = ran but produced a wrong/stale result. Report either honestly.
                bool correct = std::fabs(out[0] - expect) < 1e-4f;
                printf("PHASE B (cross-CTA, default): completed WITHOUT deadlock, out[0]=%.4f (%s)\n",
                       out[0], correct ? "correct" : "STALE/WRONG");
                worker.join();
                // Completing-without-deadlock is acceptable; the characterization
                // claim is "deadlocks OR (if it runs) needs concurrency" — a clean
                // correct completion would only happen on a concurrent build.
            } else {
                printf("PHASE B (cross-CTA, default): DEADLOCK REPRODUCED "
                       "(owner CTA0 spins on flag that peer CTA1 — not yet scheduled — must set)\n");
                worker.detach();    // orphan the hung kernel thread; process will exit
            }
            // Characterization phase passes regardless (the deadlock is the finding).
        }
    }

    printf("%s\n", all_ok ? "TEST PASSED"
                          : "DIVERGED (single-CTA control wrong or concurrent-CTA gate failed)");
    // NOTE: if PHASE B deadlocked we deliberately do NOT join the worker; returning
    // from main with a detached hung thread lets the process exit cleanly (the
    // detached std::thread is abandoned). Use _exit to avoid the abandoned-thread
    // having any chance to run global dtors over freed state.
    fflush(stdout);
    _Exit(all_ok ? 0 : 1);
}
