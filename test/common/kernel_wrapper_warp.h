// kernel_wrapper_warp.h — serialized-warp launch (--lifter x86warp).
//
// Runs a whole CTA's threads as cooperative ucontext FIBERS inside ONE OS thread
// (vs kernel_wrapper.h's one-OS-thread-per-CUDA-thread). This makes all
// cross-lane state (shared mem, mbarriers, uniform-register slots) single-
// threaded and race-free, and lets warp-uniform / lockstep semantics be modelled
// correctly.
//
// MILESTONE (i): launch + per-lane const_mem/SR/threadctx setup, fibers run to
// completion. Kernels with NO yielding collectives (e.g. vecadd) are correct here
// because each fiber sets up its own const_mem at entry and (without a yield) runs
// to completion before the next — no mid-run save/restore needed. Yielding
// collectives (mbarrier/named-barrier) + per-fiber const/local save-restore are
// milestone (ii); a fiber that needs to yield calls warp_fiber_yield().
//
// Include this INSTEAD of relying on launchKernel; it pulls in kernel_wrapper.h
// for all the lifted-code runtime (const_mem, intrinsics, contexts) and adds
// launchKernel_warp beside the untouched launchKernel.
#pragma once
#include "kernel_wrapper.h"
#include <ucontext.h>
#include <vector>
#include <functional>

namespace wfiber {

struct Fiber {
    ucontext_t ctx;
    std::vector<char> stack;
    bool started = false;
    bool exited  = false;
    bool parked  = false;     // yielded at a collective (milestone ii)
    int  tid     = 0;
    // Per-lane state saved across a yield (const_mem/local_mem/g_threadCtx/
    // g_laneHist are thread_local = one instance per OS thread, shared by all the
    // CTA's fibers; the scheduler swaps the active lane's copy in/out).
    std::vector<uint8_t> save_const;   // sizeof(const_mem)
    std::vector<uint8_t> save_local;   // sizeof(local_mem)
    ThreadCtx   save_ctx{};
    LaneHistory save_hist{};
};

struct CtaSched {
    ucontext_t sched;
    std::vector<Fiber> fib;
    int cur = -1;
    int blockDim = 0;
    int cta = 0;
    int gridDim = 0;
    std::function<void()> body;   // the lifted kernel func()
};

static thread_local CtaSched* g_sched = nullptr;

// Optional 3D grid mapping for launchKernel_warp. When g_warpGrid3D.x > 0, the
// flat cta index (0..gridDim-1) is decomposed into (blockIdx.x,y,z) row-major as
//   x = cta % gx ;  y = (cta / gx) % gy ;  z = cta / (gx*gy)
// and SR_GRID_DIM_{X,Y,Z} are set to (gx,gy,gz). Default (gx==0) preserves the
// original 1D behavior exactly (blockIdx = {cta,0,0}, gridDim = {gridDim,1,1}).
// Set by a driver BEFORE launchKernel_warp; kernels that read SR_CTAID.Y/Z (e.g.
// FA3 split-KV's SingleTileScheduler grid (m_block, split*head, batch)) need this.
struct WarpGrid3D { int x = 0, y = 1, z = 1; };
static WarpGrid3D g_warpGrid3D;

// Save / restore a lane fiber's thread_local state across a context switch.
static inline void fiber_save(Fiber& f) {
    std::memcpy(f.save_const.data(), const_mem, sizeof(const_mem));
    std::memcpy(f.save_local.data(), local_mem, sizeof(local_mem));
    f.save_ctx  = g_threadCtx;
    f.save_hist = g_laneHist;
}
static inline void fiber_restore(Fiber& f) {
    std::memcpy(const_mem, f.save_const.data(), sizeof(const_mem));
    std::memcpy(local_mem, f.save_local.data(), sizeof(local_mem));
    g_threadCtx = f.save_ctx;
    g_laneHist  = f.save_hist;
}

// A lane fiber yields back to the CTA scheduler (used by yielding collectives,
// milestone ii). Marks parked; the scheduler resumes another runnable fiber.
inline void warp_fiber_yield() {
    if (!g_sched) return;                       // not in fiber mode
    // Deferred-flip publish point: before this fiber relinquishes the OS thread,
    // flip any expect-tx mbarriers it armed (slifter_mbar_arrive_tx). The fiber
    // ran straight-line from each such arrive through its SYNCHRONOUS bulk copies
    // (UTMALDG/UBLKCP) with no yield in between, so the tiles are now published
    // in smem — only NOW may a consumer observe the phase flip and read them.
    // (No-op unless this fiber has pending expect-tx arrives; plain count-arrive
    // barriers flip eagerly in slifter_mbar_arrive and are unaffected.)
    slifter_mbar_flush_pending();
    Fiber& f = g_sched->fib[g_sched->cur];
    f.parked = true;
    swapcontext(&f.ctx, &g_sched->sched);
}
// True iff currently executing inside a serialized-warp fiber (collectives use
// this to choose yield vs the per-thread condvar path).
inline bool in_warp_fiber() { return g_sched != nullptr; }

// Per-fiber entry: set up this lane's const_mem/SR/threadctx exactly like the
// per-thread launchKernel path, then run the lifted kernel.
static void fiber_entry() {
    CtaSched* S = g_sched;
    int tid = S->fib[S->cur].tid;
    int cta = S->cta, blockDim = S->blockDim, gridDim = S->gridDim;

    std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));
    std::memset(local_mem, 0, sizeof(local_mem));
    uint8_t* c0 = cmem0();
    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X)     = (uint32_t)blockDim;
    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Y)     = 1u;
    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Z)     = 1u;
    // 3D grid mapping (opt-in via g_warpGrid3D; default preserves 1D behavior).
    uint32_t bidx = (uint32_t)cta, bidy = 0u, bidz = 0u;
    uint32_t gdx = (uint32_t)gridDim, gdy = 1u, gdz = 1u;
    if (g_warpGrid3D.x > 0) {
        gdx = (uint32_t)g_warpGrid3D.x; gdy = (uint32_t)g_warpGrid3D.y; gdz = (uint32_t)g_warpGrid3D.z;
        bidx = (uint32_t)(cta % g_warpGrid3D.x);
        bidy = (uint32_t)((cta / g_warpGrid3D.x) % g_warpGrid3D.y);
        bidz = (uint32_t)(cta / (g_warpGrid3D.x * g_warpGrid3D.y));
    }
    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = gdx;
    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Y) = gdy;
    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Z) = gdz;
    *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X)    = bidx;
    *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_Y)    = bidy;
    *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_Z)    = bidz;
    *reinterpret_cast<uint32_t*>(c0 + SR_TID_X)      = (uint32_t)tid;
    *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID)    = (uint32_t)(tid & (WARP_SIZE - 1));

    g_threadCtx.cta         = cta;
    g_threadCtx.tid         = tid;
    g_threadCtx.lane        = tid & (WARP_SIZE - 1);
    g_threadCtx.warpInBlock = tid / WARP_SIZE;
    g_threadCtx.warpGlobal  = cta * g_warpsPerBlock + g_threadCtx.warpInBlock;

    S->body();
    // A producer fiber that ends without a final yield (its last expect-tx arrive
    // + bulk copies, then EXIT) must still publish those deferred phase flips so a
    // consumer waiting on the last tile is released. Flush here while g_threadCtx
    // still belongs to this fiber (the pending set is per-fiber thread_local).
    slifter_mbar_flush_pending();
    S->fib[S->cur].exited = true;
    swapcontext(&S->fib[S->cur].ctx, &S->sched);   // back to scheduler, done
}

} // namespace wfiber

// -----------------------------------------------------------------------------
//   launchKernel_warp(func, gridDim, blockDim, args...)
// CTAs run sequentially (per-CTA shared_mem). Within a CTA, blockDim lanes run
// as cooperative fibers in this OS thread.
// -----------------------------------------------------------------------------
template<typename Func, typename... Args>
void launchKernel_warp(Func func, int gridDim, int blockDim, Args... args)
{
    off = ARG_BASE;
    (writeArg(args), ...);
    std::memcpy(g_constMemTemplate.data(), const_mem, sizeof(const_mem));

    g_blockDim      = blockDim;
    g_gridDim       = gridDim;
    g_warpsPerBlock = (blockDim + WARP_SIZE - 1) / WARP_SIZE;
    const int totalWarps = g_warpsPerBlock * gridDim;

    // Reuse the per-warp / per-block context arrays (collectives still key off
    // them; under fibers they are touched by one OS thread so no locking races).
    std::vector<WarpShuffleCtx>  warpCtx(totalWarps);
    std::vector<WarpVoteCtx>     voteCtx(totalWarps);
    std::vector<WarpVoteAllCtx>  voteAllCtx(totalWarps);
    std::vector<WarpHistCtx>     histCtx(totalWarps);
    std::vector<WarpGroupCtx>    groupCtx(totalWarps);
    std::vector<BlockSyncCtx>    blockCtx(gridDim);
    // Named/partial barrier state (BAR.SYNC/BAR.ARV <id>,<count>): one generation
    // barrier per (CTA, barrier id). Default-initialized (arrived=0, gen=0); each
    // CTA uses its own cta*16 slice and CTAs run sequentially, so no reset needed.
    std::vector<NamedBarCtx>     namedBar((size_t)gridDim * SLIFTER_MAX_NAMED_BAR);
    // Per-(CTA, warpgroup) barrier for the RS-form WGMMA A-fragment spill (FA PV GEMM).
    std::vector<NamedBarCtx>     rsBar((size_t)gridDim * SLIFTER_MAX_NAMED_BAR);
    g_warpCtx = warpCtx.data(); g_warpVoteCtx = voteCtx.data();
    g_warpVoteAllCtx = voteAllCtx.data(); g_warpHistCtx = histCtx.data();
    g_warpGroupCtx = groupCtx.data(); g_blockSyncCtx = blockCtx.data();
    g_namedBarCtx = namedBar.data();
    g_rsBarCtx = rsBar.data();
    // The matrix/movmatrix collective contexts the per-thread launch also sets
    // (STSM/HMMA/LDSM/DMMA/IMMA/MOVM) — FA3's WGMMA epilogue uses stsm_m88_warp,
    // which locks g_warpStsmCtx; without these the collectives deref a null ctx.
    std::vector<WarpHmmaCtx>     hmmaCtx(totalWarps);
    std::vector<WarpLdsmCtx>     ldsmCtx(totalWarps);
    std::vector<WarpStsmCtx>     stsmCtx(totalWarps);
    std::vector<WarpDmmaCtx>     dmmaCtx(totalWarps);
    std::vector<WarpDmma884Ctx>  dmma884Ctx(totalWarps);
    std::vector<WarpImma8816Ctx> imma8816Ctx(totalWarps);
    std::vector<WarpImma16832Ctx> imma16832Ctx(totalWarps);
    std::vector<WarpImmaSp16864Ctx> immaSp16864Ctx(totalWarps);
    std::vector<WarpImmaSpS4Ctx> immaSpS4Ctx(totalWarps);
    std::vector<WarpMovmCtx>     movmCtx(totalWarps);
    g_warpHmmaCtx = hmmaCtx.data(); g_warpLdsmCtx = ldsmCtx.data();
    g_warpStsmCtx = stsmCtx.data(); g_warpDmmaCtx = dmmaCtx.data();
    g_warpDmma884Ctx = dmma884Ctx.data(); g_warpImma8816Ctx = imma8816Ctx.data();
    g_warpImma16832Ctx = imma16832Ctx.data();
    g_warpImmaSp16864Ctx = immaSp16864Ctx.data();
    g_warpImmaSpS4Ctx = immaSpS4Ctx.data(); g_warpMovmCtx = movmCtx.data();

    // Arm the serialized-warp collective hooks: in-fiber collectives YIELD
    // instead of condvar-blocking (see kernel_wrapper.h warp_fiber_active()).
    g_warp_in_fiber = &wfiber::in_warp_fiber;
    g_warp_yield    = &wfiber::warp_fiber_yield;

    static const uint64_t initialRing[1] = {0xcbf29ce484222325ull};
    for (int cta = 0; cta < gridDim; ++cta) {
        std::memset(shared_mem, 0, SLIFTER_SHM_PAYLOAD_BYTES);
        // Full-CTA barrier (syncthreads) participant count for the fiber path.
        blockCtx[cta].live = blockDim;
        blockCtx[cta].arrived = 0;
        blockCtx[cta].generation = 0;
        // Pre-publish every lane's initial gather history + set live counts, so
        // warp_group_gather (VOTE/SHFL/MATCH) does NOT see not-yet-scheduled
        // fibers as EXITED (ring==nullptr). Mirrors the per-thread launch path;
        // required because fibers run one-at-a-time, not all-spawned-at-once.
        for (int w = 0; w < g_warpsPerBlock; ++w) {
            const int wg = cta * g_warpsPerBlock + w;
            const int lanes = std::min(WARP_SIZE, blockDim - w * WARP_SIZE);
            warpCtx[wg].live = lanes; voteCtx[wg].live = lanes; voteAllCtx[wg].live = lanes;
            hmmaCtx[wg].live = lanes; ldsmCtx[wg].live = lanes; stsmCtx[wg].live = lanes;
            dmmaCtx[wg].live = lanes; dmma884Ctx[wg].live = lanes;
            imma8816Ctx[wg].live = lanes; imma16832Ctx[wg].live = lanes;
            immaSp16864Ctx[wg].live = lanes; immaSpS4Ctx[wg].live = lanes;
            movmCtx[wg].live = lanes;
            for (int l = 0; l < lanes; ++l) {
                histCtx[wg].seq[l]  = 0;
                histCtx[wg].hash[l] = 0xcbf29ce484222325ull;
                histCtx[wg].ring[l] = initialRing;
            }
        }

        wfiber::CtaSched S;
        S.blockDim = blockDim; S.cta = cta; S.gridDim = gridDim;
        S.body = [&](){ func(); };
        S.fib.resize(blockDim);
        for (int t = 0; t < blockDim; ++t) {
            S.fib[t].tid = t;
            S.fib[t].stack.resize(8 * 1024 * 1024);  // 8MB: -O0 lifted kernels have huge frames
            S.fib[t].save_const.resize(sizeof(const_mem));
            S.fib[t].save_local.resize(sizeof(local_mem));
            getcontext(&S.fib[t].ctx);
            S.fib[t].ctx.uc_stack.ss_sp = S.fib[t].stack.data();
            S.fib[t].ctx.uc_stack.ss_size = S.fib[t].stack.size();
            S.fib[t].ctx.uc_link = &S.sched;
            makecontext(&S.fib[t].ctx, wfiber::fiber_entry, 0);
        }
        wfiber::g_sched = &S;

        // Cooperative scheduler: round-robin all non-exited fibers. A fiber that
        // PARKS (yielded at a yield-and-repoll collective) is resumed next pass —
        // collectives self-resolve via shared state + re-poll, so the scheduler
        // needs no explicit rendezvous logic. Per-lane thread_local state is
        // saved on park and restored before resume (it aliases across the CTA's
        // fibers in this one OS thread). Milestone (i) kernels never park.
        long guard = 0;
        bool any = true;
        while (any) {
            any = false;
            int progressed = 0;
            for (int t = 0; t < blockDim; ++t) {
                if (S.fib[t].exited) continue;
                any = true;
                if (S.fib[t].started) wfiber::fiber_restore(S.fib[t]);  // swap this lane in
                S.cur = t; S.fib[t].parked = false;
                swapcontext(&S.sched, &S.fib[t].ctx);
                S.fib[t].started = true;
                if (!S.fib[t].exited) { wfiber::fiber_save(S.fib[t]); progressed++; } // parked: save state
                // Runaway = no fiber reached exit within 200M swaps: a lifted-
                // synchronization deadlock/livelock (e.g. an mbarrier that never
                // flips). Abort LOUDLY — returning here would let the launch
                // "complete" with partial state and surface later as a silent
                // numeric mismatch (or worse, a fake pass on untouched outputs).
                if (++guard > 200000000L) {
                    std::fprintf(stderr,
                        "FATAL: warp-fiber scheduler runaway (cta=%d): >200M fiber "
                        "swaps without CTA completion — lifted synchronization "
                        "deadlock/livelock. Aborting.\n", cta);
                    std::abort();
                }
            }
            (void)progressed;
        }
        wfiber::g_sched = nullptr;
    }
}
