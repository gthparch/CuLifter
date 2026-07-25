#pragma once

// -----------------------------------------------------------------------------
// Runtime support for lifted CUDA SASS (SM75)
// - const_mem  : per-thread "constant memory" (c[0]..c[4])
// - shared_mem : per-block shared memory (single segment, reused per CTA)
// - local_mem  : per-thread local memory
// - warp_shfl / vote_any / syncthreads
// - launchKernel: runs a lifted kernel on CPU threads
// -----------------------------------------------------------------------------

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <bit>
#include <cmath>
#include <iostream>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <array>
#include <algorithm>
#include <atomic>
#include <unordered_map>
#include <set>
#include <typeinfo>
#include <chrono>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <csignal>
#include <unistd.h>

constexpr int WARP_SIZE = 32;

static inline std::string slifter_debug_path(const char* filename)
{
    const char* directory = std::getenv("SLIFTER_DEBUG_DIR");
    if (!directory || !*directory || std::strcmp(directory, ".") == 0)
        return filename;
    std::string path(directory);
    if (path.back() != '/')
        path.push_back('/');
    path += filename;
    return path;
}

// -----------------------------------------------------------------------------
// Global memory regions matching lifted IR symbols
// -----------------------------------------------------------------------------
//
// IR expects (declared sizes are nominal — the lifter emits an EXTERNAL
// declaration, so this definition may be larger; GEP offsets are dynamic):
//
//   @"const_mem"  = external thread_local global [5 x [4096 x i8]]
//   @"shared_mem" = external global              [32768 x i8]
//   @"local_mem"  = external thread_local global [32768 x i8]
//
// We model:
//
//   - const_mem: thread_local (per CUDA thread)
//   - shared_mem: one global segment reused per CTA (we run CTAs
//     sequentially), sized to the sm_90 carveout + PROT_NONE guard (below)
//   - local_mem: thread_local (per CUDA thread)
// -----------------------------------------------------------------------------

// const_mem[0] corresponds to c[0], used for params and "special regs".
// NOTE: defined inside extern "C" { } block (not as 'extern "C" T x = init')
// to avoid the -Wextern-initializer diagnostic that some g++ versions treat as
// an error (observed on GH200 with Ubuntu g++ 11
// 27/28 link_fail — same warning as build error; fixed here so -Werror-safe).
// Shared-memory sizing: the PAYLOAD is the sm_90 per-SM shared carveout
// (228 KB) — the largest shared extent any supported arch can grant one CTA
// (sm_75: 64KB, sm_80: 164KB, sm_90/sm_100: 228KB incl. the 1KB HW-reserve).
// The wrapper is arch-generic, so it provisions the max: a launch config
// whose shared footprint fits the real GPU also fits here (the old 192KB
// array under-provisioned sm_90 — a 200KB-class config silently wrote past
// the array into the host heap). Accesses BEYOND the carveout are impossible
// on every supported GPU (the CUDA driver would refuse the launch), so the
// GUARD pages after the payload are mprotect'ed PROT_NONE and the fault
// handler below reports SKIP_SHMEM instead of letting an over-shared config
// corrupt the host silently (bn_oracle audit: cudnn bn_bw_1C11_singleread
// with countPerChannel=32768 caches ~263KB of x/dy in shared — heap
// corruption presented as flaky SEGV/garbage "pass").
constexpr uint32_t SLIFTER_SHM_PAYLOAD_BYTES = 228u * 1024u;  // 57 pages
constexpr uint32_t SLIFTER_SHM_GUARD_BYTES   = 16u * 1024u;   // PROT_NONE

extern "C" {
alignas(8) thread_local uint8_t const_mem[5][4096] = {0};

// ONE shared segment reused for each CTA. We run CTAs sequentially to give
// per-block semantics even though IR has a single global symbol. The tail
// SLIFTER_SHM_GUARD_BYTES are the protected guard — never readable/writable;
// only the payload may be touched (memset payload-only below).
alignas(4096) int32_t shared_mem[(SLIFTER_SHM_PAYLOAD_BYTES +
                                  SLIFTER_SHM_GUARD_BYTES) / 4] = {0};

// per-thread "local" memory
alignas(16) thread_local uint8_t local_mem[32768] = {0};
} // extern "C"

// -----------------------------------------------------------------------------
// Shared-memory overflow guard (SKIP_SHMEM)
//
// Lifted LDS/STS are unbounded GEPs into shared_mem; a generated launch
// config whose shared footprint exceeds the sm_90 carveout used to scribble
// past the array into unrelated globals/heap — a garbage run that still
// printed TEST PASSED. The guard converts exactly that class into an
// explicit, deterministic verdict: a fault inside the guard pages prints
// "SKIP_SHMEM: ..." (async-signal-safe) and exits 4; the suite runner maps
// it to status `skip_shmem` (config invalid on real hardware too — the CUDA
// driver would have refused the launch). Faults anywhere else re-execute
// under SIG_DFL so genuine crashes keep their honest signal verdict.
// -----------------------------------------------------------------------------

namespace slifter_shm_guard {

inline int u64_to_dec(uint64_t v, char* out) {
    char tmp[20];
    int n = 0;
    do { tmp[n++] = static_cast<char>('0' + v % 10); v /= 10; } while (v);
    for (int i = 0; i < n; ++i) out[i] = tmp[n - 1 - i];
    return n;
}

inline void fault_handler(int sig, siginfo_t* si, void*) {
    const uintptr_t base = reinterpret_cast<uintptr_t>(shared_mem);
    const uintptr_t g0 = base + SLIFTER_SHM_PAYLOAD_BYTES;
    const uintptr_t g1 = g0 + SLIFTER_SHM_GUARD_BYTES;
    const uintptr_t a = reinterpret_cast<uintptr_t>(si ? si->si_addr : nullptr);
    if (a >= g0 && a < g1) {
        char msg[192];
        int n = 0;
        static const char head[] = "SKIP_SHMEM: shared-memory access at byte offset ";
        for (const char* p = head; *p; ++p) msg[n++] = *p;
        n += u64_to_dec(a - base, msg + n);
        static const char tail[] = " exceeds the 233472-byte sm_90 carveout;"
                                   " launch config cannot run on hardware\n";
        for (const char* p = tail; *p; ++p) msg[n++] = *p;
        ssize_t wr = write(1, msg, static_cast<size_t>(n));
        (void)wr;
        _exit(4);
    }
    // Not the shared guard: restore the default disposition and return; the
    // faulting instruction re-executes and the process dies with the genuine
    // signal, so runner crash classification is unchanged.
    std::signal(sig, SIG_DFL);
}

struct Init {
    Init() {
        ::mprotect(reinterpret_cast<uint8_t*>(shared_mem) +
                       SLIFTER_SHM_PAYLOAD_BYTES,
                   SLIFTER_SHM_GUARD_BYTES, PROT_NONE);
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = fault_handler;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        ::sigaction(SIGSEGV, &sa, nullptr);
        ::sigaction(SIGBUS, &sa, nullptr);
    }
};
static Init g_init;

} // namespace slifter_shm_guard

// Helper: base of c[0]
inline uint8_t* cmem0() {
    return &const_mem[0][0];
}

// -----------------------------------------------------------------------------
// Special-register layout in c[0]
// -----------------------------------------------------------------------------

enum : uint32_t {
    SR_NTID_X     = 0x0,
    SR_NTID_Y     = 0x4,
    SR_NTID_Z     = 0x8,
    SR_GRID_DIM_X = 0xC,
    SR_GRID_DIM_Y = 0x10,
    SR_GRID_DIM_Z = 0x14,
    SR_CTAID_X    = 0x20,
    SR_CTAID_Y    = 0x24,
    SR_CTAID_Z    = 0x28,
    SR_TID_X      = 0x2C,
    SR_TID_Y      = 0x30,
    SR_TID_Z      = 0x34,
    SR_LANE_ID    = 0x38,
#ifndef SLIFTER_ARG_BASE
    ARG_BASE      = 0x160
#else
    ARG_BASE      = SLIFTER_ARG_BASE
#endif
};

static uint32_t off = ARG_BASE;

static constexpr uint32_t align_up(uint32_t x, uint32_t a) {
    return (x + (a - 1u)) & ~(a - 1u);
}

// -----------------------------------------------------------------------------
// Argument marshalling into c[0][ARG_BASE...]
// -----------------------------------------------------------------------------

template<typename T>
inline void writeArg(T v)
{
    uint32_t aligned_off = align_up(off, static_cast<uint32_t>(sizeof(T)));
    uint8_t* base = cmem0();

    if constexpr (std::is_pointer_v<T>) {
        uint64_t raw = reinterpret_cast<uint64_t>(v);
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " ptr=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw = ({ uint64_t _r; memcpy(&_r, &v, 8); _r; });
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " u64=0x" << std::hex << raw << std::dec;
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw = ({ uint32_t _r; memcpy(&_r, &v, 4); _r; });
        std::memcpy(base + aligned_off, &raw, sizeof(raw));
        std::cout << " u32=0x" << std::hex << raw << std::dec;
    } else {
        // BUG-051: Support arbitrary sizes (e.g. bool=1 byte, short=2 bytes).
        // Use memcpy for type-punning safety.
        std::memcpy(base + aligned_off, &v, sizeof(T));
        std::cout << " raw[" << sizeof(T) << "]";
    }

    std::cout << " Arg byte_off=" << aligned_off << ", type=" << typeid(T).name()
              << ", size=" << sizeof(T) << std::endl;

    off = aligned_off + sizeof(T);
}

// Explicit-offset pin. Generated drivers emit ArgAt{off} before a param when
// the ELF's EIATTR_KPARAM_INFO provides the authoritative const-bank offset,
// so writeArg's sizeof-based alignment can never drift from the real ABI
// layout (by-value structs were the failure mode: an 8-byte struct guessed
// as a 4-byte int shifted every later param).
struct ArgAt { uint32_t abs_off; };
inline void writeArg(ArgAt a) { off = a.abs_off; }

// -----------------------------------------------------------------------------
// Per-thread context (for shuffles, votes, syncthreads)
// -----------------------------------------------------------------------------

struct ThreadCtx {
    int cta         = 0;
    int tid         = 0;
    int lane        = 0;
    int warpInBlock = 0;
    int warpGlobal  = 0;
};

thread_local ThreadCtx g_threadCtx;

// Serialized-warp (x86warp) hooks. kernel_wrapper_warp.h sets these to its fiber
// scheduler; when null (the per-CUDA-thread x86/arm path) the collectives use
// their normal condvar blocking, byte-identical to before. When set, a collective
// that would block instead YIELDS to the CTA's fiber scheduler (a condvar block
// would freeze the whole CTA's single OS thread).
static bool (*g_warp_in_fiber)() = nullptr;
static void (*g_warp_yield)()    = nullptr;
static inline bool warp_fiber_active() { return g_warp_in_fiber && g_warp_in_fiber(); }

// -----------------------------------------------------------------------------
// Spin diagnostics (opt-in: SLIFTER_SPIN_DEBUG=1)
//
// A collective wait storm has two distinct shapes, probed separately:
//  - SPIN: one cv-wait blocked too long — a peer is missing from the
//    rendezvous (exited early / predicated off / diverged).
//  - HOTCALL: a barrier or collective COMPLETING over and over — the
//    kernel-level loop around it (e.g. CUTLASS Semaphore::wait polling a
//    lock word) never converges. Each iteration looks healthy in isolation.
// Zero overhead when the env var is unset: one cached static check per site.
// -----------------------------------------------------------------------------

static inline bool spin_debug_on()
{
    static const bool en = std::getenv("SLIFTER_SPIN_DEBUG") != nullptr;
    return en;
}

// cv.wait(lock, pred) that additionally reports ONCE (per call) when blocked
// longer than ~5s. `report` runs under the caller's lock. Semantics match
// cv.wait(lock, pred): pred is only evaluated with the lock held.
template <typename Pred, typename Report>
inline void spin_diag_wait(std::condition_variable& cv,
                           std::unique_lock<std::mutex>& lock,
                           Pred pred, Report report)
{
    if (!spin_debug_on()) { cv.wait(lock, pred); return; }
    const auto t0 = std::chrono::steady_clock::now();
    bool reported = false;
    while (!pred()) {
        cv.wait_for(lock, std::chrono::milliseconds(100));
        if (!reported &&
            std::chrono::steady_clock::now() - t0 > std::chrono::seconds(5)) {
            reported = true;
            report();
        }
    }
}

// Per-lane futex wait/wake: the gather decide wait sleeps directly on its
// own ready word, so the decisive inserter wakes exactly the members of the
// decided instance. Sleeping decide-waiters on the per-warp condition
// variable instead meant every stamp/sweep notify_all woke all ~31 sleepers
// through the still-held mutex (~190 context-switch pairs per gather; with
// CB_strsm's ~1.2M gathers that was a ~900k switches/s futex storm and tens
// of CPU-minutes of sys time).
static inline void futex_wait_u32(std::atomic<uint32_t>* a, uint32_t expect,
                                  long timeout_ms)
{
    struct timespec ts{timeout_ms / 1000, (timeout_ms % 1000) * 1000000L};
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(a), FUTEX_WAIT_PRIVATE,
            expect, &ts, nullptr, 0);
}

static inline void futex_wake_u32(std::atomic<uint32_t>* a, int n)
{
    syscall(SYS_futex, reinterpret_cast<uint32_t*>(a), FUTEX_WAKE_PRIVATE,
            n, nullptr, nullptr, 0);
}

// Call-count probe: returns the count when this site's per-thread call count
// hits a power of two >= 2^10 (caller prints its own line with state), else 0.
// A barrier completing ~50x/s with 256 threads crosses 2^10 in ~20s — well
// inside a run window — while normal kernels stay far below it.
static inline uint64_t spin_hotcall(const char* site)
{
    if (!spin_debug_on()) return 0;
    constexpr int N = 8;
    struct Slot { const char* site; uint64_t n; };
    thread_local Slot tab[N] = {};
    for (int i = 0; i < N; ++i) {
        if (tab[i].site == site || tab[i].site == nullptr) {
            tab[i].site = site;
            const uint64_t n = ++tab[i].n;
            return (n >= (1ull << 10) && (n & (n - 1)) == 0) ? n : 0;
        }
    }
    return 0;
}

// -----------------------------------------------------------------------------
// Divergence history (warp-model stage 2, shadow mode)
// BSSY/BSYNC lift to these events on the CPU path; the rolling hash gives
// each lane a path signature that distinguishes loop iterations and branch
// nesting. Shadow mode: collectives publish + compare signatures and report
// (once) when an arrival-based group mixes lanes with different histories —
// validation data for switching membership to history matching.
// -----------------------------------------------------------------------------

// Ring of this lane's recent path hashes, indexed by event sequence number.
// Another lane B with published (seq_B, hash_B) is ON MY PATH (just behind)
// iff my ring at seq_B holds hash_B; otherwise our paths diverged at or
// before seq_B. Ring capacity bounds how far behind a lane may lag while
// still being recognized — 4096 events is far beyond any loop body between
// two collectives in practice.
constexpr int HIST_RING = 4096;

struct LaneHistory {
    uint64_t hash  = 0xcbf29ce484222325ull;   // FNV offset basis
    uint32_t depth = 0;
    uint64_t seq   = 0;
    uint64_t ring[HIST_RING];
    LaneHistory() { ring[0] = hash; }
};
thread_local LaneHistory g_laneHist;

// Per-warp published view of every lane's (seq, hash) + its prefix ring,
// updated on every region event so waiting collectives can classify peers.
struct WarpHistCtx {
    std::mutex m;
    uint64_t seq[WARP_SIZE]{};
    uint64_t hash[WARP_SIZE]{};
    const uint64_t* ring[WARP_SIZE]{};   // points at the lane's TLS ring
    // Lane parked at a block barrier: not in any collective's activemask
    // while waiting there (legal CUDA cannot interleave a block barrier
    // with a warp collective of the same lanes).
    bool parked[WARP_SIZE]{};
};
static WarpHistCtx* g_warpHistCtx = nullptr;
static void warp_hist_publish();

extern "C" void warp_region_push(uint32_t site)
{
    g_laneHist.hash = (g_laneHist.hash ^ (uint64_t(site) | 0x100000000ull)) * 0x100000001b3ull;
    g_laneHist.depth++;
    g_laneHist.seq++;
    g_laneHist.ring[g_laneHist.seq % HIST_RING] = g_laneHist.hash;
    warp_hist_publish();
}

extern "C" void warp_region_pop(uint32_t site)
{
    g_laneHist.hash = (g_laneHist.hash ^ (uint64_t(site) | 0x200000000ull)) * 0x100000001b3ull;
    if (g_laneHist.depth) g_laneHist.depth--;
    g_laneHist.seq++;
    g_laneHist.ring[g_laneHist.seq % HIST_RING] = g_laneHist.hash;
    warp_hist_publish();
}

// A block barrier (__syncthreads / BAR.SYNC / BAR.RED) is a full reconvergence
// point: HW requires every live thread to arrive, so on the far side the warp is
// fully converged and the next warp collective sees ALL its lanes. The per-lane
// divergence-history (seq, hash) must therefore be RESYNCHRONIZED across the warp
// here, or a partial-warp collective taken before the barrier (e.g. a block
// reduce's 8-lane partial sum, executed by only lanes 0..7) leaves those lanes at
// a higher event count than their siblings, and a SUBSEQUENT full-warp shuffle
// splits into disjoint history groups — silently dropping the non-matching lanes'
// contributions (so.193 softmax sum-reduction lost ~1.5% mass: warp0 split
// {0..7}/{8..31}). `epoch` is the post-barrier block generation: monotonic and
// identical for every lane of the warp that passed the same barrier instance, so
// all lanes land on one (seq, hash) and re-match. depth resets to the common
// (reconverged) region. The 2^20 seq stride leaves ample headroom for any
// collectives a lane runs before the next barrier while staying warp-uniform.
static void warp_hist_resync_after_barrier(uint64_t epoch)
{
    g_laneHist.depth = 0;
    g_laneHist.seq   = epoch << 20;
    g_laneHist.hash  = (epoch + 0x9E3779B97F4A7C15ull) * 0x100000001b3ull;
    g_laneHist.ring[g_laneHist.seq % HIST_RING] = g_laneHist.hash;
    warp_hist_publish();
}

static std::atomic<int> g_shadowMismatchReports{0};

// -----------------------------------------------------------------------------
// Per-warp contexts for shuffles and votes
// -----------------------------------------------------------------------------

struct WarpShuffleCtx {
    std::mutex m;
    std::condition_variable cv;
    // Lanes of this warp still executing (HW: EXITed lanes leave the
    // activemask). Collectives complete when every LIVE lane arrived.
    int   live       = 0;
    int   arrived    = 0;
    int   generation = 0;
    // Raw 32-bit payload; we bit_cast between int32/float and this.
    uint32_t vals[WARP_SIZE]{};
    uint32_t results[WARP_SIZE]{};
};

struct WarpVoteCtx {
    std::mutex m;
    std::condition_variable cv;
    // Lanes of this warp still executing (HW: EXITed lanes leave the
    // activemask). Collectives complete when every LIVE lane arrived.
    int   live       = 0;
    uint64_t laneHist[WARP_SIZE]{};   // shadow: per-lane divergence hashes
    int      arrived    = 0;
    int      generation = 0;
    bool     any        = false;
    bool     result     = false;
    bool     preds[WARP_SIZE]{};   // per-lane predicate for ballot
    uint32_t ballotMask = 0;       // computed ballot bitmask
};

struct WarpVoteAllCtx {
    std::mutex m;
    std::condition_variable cv;
    // Lanes of this warp still executing (HW: EXITed lanes leave the
    // activemask). Collectives complete when every LIVE lane arrived.
    int   live       = 0;
    int  arrived    = 0;
    int  generation = 0;
    bool all        = true;
    bool result     = false;
};

// -----------------------------------------------------------------------------
// Per-block context for syncthreads
// -----------------------------------------------------------------------------

struct BlockSyncCtx {
    std::mutex m;
    std::condition_variable cv;
    int arrived    = 0;
    int generation = 0;
    // Threads still able to reach the barrier. On HW, EXIT removes a thread
    // from barrier participation; a thread that returns before __syncthreads
    // (e.g. the `if (gid >= n) return;` guard with blockDim > n) must not be
    // waited for, or every remaining thread deadlocks.
    int live       = 0;
    // BAR.RED.{AND,OR,POPC} predicate-reduction state for barrier 0.
    // Accumulated under `m` by arriving barrier0_* callers; snapshot to
    // *_res and reset at the generation flip by WHOEVER completes the
    // barrier (syncthreads, barrier0_*, or a thread exit) — HW lets
    // BAR.SYNC and BAR.RED arrivals at the same barrier complete together.
    // A waiter's *_res read after wake is race-free: the NEXT flip needs
    // all live threads to arrive again, impossible while one is still
    // inside this call.
    int red_and  = 1;
    int red_or   = 0;
    int red_popc = 0;
    int red_and_res  = 1;
    int red_or_res   = 0;
    int red_popc_res = 0;
};

// Global execution config visible to primitives
static WarpShuffleCtx*  g_warpCtx         = nullptr;
static WarpVoteCtx*     g_warpVoteCtx     = nullptr;
static WarpVoteAllCtx*  g_warpVoteAllCtx  = nullptr;
static BlockSyncCtx*    g_blockSyncCtx    = nullptr;
static int             g_blockDim      = 0;
static int             g_gridDim       = 0;
static int             g_warpsPerBlock = 0;

// -----------------------------------------------------------------------------
// Lockstep turnstile (warp-model stage 3, opt-in)
//
// Pre-Volta warp-synchronous code (volatile shared memory, NO __syncwarp —
// the CUDPP scanwarp idiom in split-cuda and friends) is correct only under
// SIMT lockstep: every lane's k-th shared access retires before any lane's
// (k+1)-th begins. Free-running lane threads violate that (lanes read peers'
// CURRENT values where HW guarantees the previous step's).
//
// Opt-in fix, two flags:
//   - lift with CULIFTER_LOCKSTEP=1  -> IR wraps each shared access (LDS/
//     STS/ATOMS/...) in warp_turn_begin()/warp_turn_end()
//   - run  with SLIFTER_LOCKSTEP=1   -> launchKernel arms a per-warp token
// The token visits live, unparked lanes in cyclic lane order; an access
// executes only while its lane HOLDS the token (begin waits for it, end
// passes it on), so round k of accesses fully retires before round k+1.
// Lanes blocked in a barrier or collective park themselves so the token
// skips them. Either flag absent => everything below is a no-op.
//
// Faithfulness note: for CONVERGED warps this reproduces per-instruction
// SIMT ordering exactly (within one round all lanes touch distinct slots or
// all read — same-instruction concurrency is HW-simultaneous anyway). For
// divergent lanes the rounds interleave both branch sides, which HW would
// serialize — divergent unsynchronized shared choreography has no defined
// order on HW either, so this is not weaker than what such code may assume.
// A lane spinning on a flag set by a SAME-warp peer (legal only with
// independent thread scheduling, sm_70+) stalls its warp's token until the
// spin resolves; if the setter is in the same warp that construct needs ITS
// and is out of this model's scope.
// -----------------------------------------------------------------------------

struct WarpTurnCtx {
    std::mutex m;
    std::condition_variable cv;
    int  cur   = 0;                 // lane whose turn it is
    int  lanes = WARP_SIZE;         // lanes in this warp (last warp may be short)
    bool parked[WARP_SIZE]{};       // blocked in barrier/collective: skip
    bool exited[WARP_SIZE]{};       // returned from the kernel: skip forever
};
static WarpTurnCtx* g_warpTurnCtx = nullptr;   // null => mode off

// Advance cur to the next claimable lane (cyclic). Caller holds ctx.m.
// If every lane is parked/exited, cur is left stale — the next unpark
// re-points it at itself.
static void warp_turn_advance_(WarpTurnCtx& c)
{
    for (int step = 1; step <= c.lanes; ++step) {
        const int l = (c.cur + step) % c.lanes;
        if (!c.parked[l] && !c.exited[l]) { c.cur = l; return; }
    }
}

extern "C" void warp_turn_begin()
{
    if (!g_warpTurnCtx) return;
    WarpTurnCtx& c = g_warpTurnCtx[g_threadCtx.warpGlobal];
    std::unique_lock<std::mutex> lk(c.m);
    spin_diag_wait(c.cv, lk, [&]{ return c.cur == g_threadCtx.lane; }, [&]{
        std::fprintf(stderr, "SPIN site=turn_begin tid=%d cta=%d lane=%d cur=%d\n",
                     g_threadCtx.tid, g_threadCtx.cta, g_threadCtx.lane, c.cur);
    });
}

extern "C" void warp_turn_end()
{
    if (!g_warpTurnCtx) return;
    WarpTurnCtx& c = g_warpTurnCtx[g_threadCtx.warpGlobal];
    { std::lock_guard<std::mutex> lk(c.m);
      if (c.cur == g_threadCtx.lane) warp_turn_advance_(c); }
    c.cv.notify_all();
}

// RAII park: removes the lane from token rotation while it blocks inside a
// wrapper primitive (block barrier, warp collective). On unpark, reclaims a
// stranded token (cur left on a parked/exited lane when everyone was parked).
struct TurnParkGuard {
    TurnParkGuard()
    {
        if (!g_warpTurnCtx) return;
        WarpTurnCtx& c = g_warpTurnCtx[g_threadCtx.warpGlobal];
        { std::lock_guard<std::mutex> lk(c.m);
          c.parked[g_threadCtx.lane] = true;
          if (c.cur == g_threadCtx.lane) warp_turn_advance_(c); }
        c.cv.notify_all();
    }
    ~TurnParkGuard()
    {
        if (!g_warpTurnCtx) return;
        WarpTurnCtx& c = g_warpTurnCtx[g_threadCtx.warpGlobal];
        { std::lock_guard<std::mutex> lk(c.m);
          c.parked[g_threadCtx.lane] = false;
          if (c.parked[c.cur] || c.exited[c.cur]) c.cur = g_threadCtx.lane; }
        c.cv.notify_all();
    }
    TurnParkGuard(const TurnParkGuard&) = delete;
    TurnParkGuard& operator=(const TurnParkGuard&) = delete;
};

static void warp_turn_exit_()
{
    if (!g_warpTurnCtx) return;
    WarpTurnCtx& c = g_warpTurnCtx[g_threadCtx.warpGlobal];
    { std::lock_guard<std::mutex> lk(c.m);
      c.exited[g_threadCtx.lane] = true;
      if (c.cur == g_threadCtx.lane) warp_turn_advance_(c); }
    c.cv.notify_all();
}

// -----------------------------------------------------------------------------
// History-matched warp group collectives (warp-model stage 2, step 2)
//
// Membership of a collective instance = live lanes whose divergence history
// matches the caller's (same rolling hash). A lane behind the caller on the
// SAME path has the caller's hash in its future; we recognize "behind on my
// path" by checking the caller's prefix ring at the peer's sequence number.
// Completion rule per caller: every live lane is either (a) arrived with my
// hash (in my group), or (b) provably not in this instance — diverged
// (their published hash is not on my path) or ahead (they passed this point
// in an earlier instance). Region events publish (seq, hash) and notify, so
// waits resolve as information arrives — never blocked on the other branch
// side's completion.
// -----------------------------------------------------------------------------

struct GroupEntry {
    bool     valid    = false;
    bool     consumed = false;
    bool     pred     = false;
    uint32_t val      = 0;
    uint64_t hash     = 0;
};

struct WarpGroupCtx {
    std::mutex m;
    std::condition_variable cv;
    GroupEntry e[WARP_SIZE];
    // ready[L]==1: the instance L's current entry belongs to was decided by
    // an insert sweep (set under m by the decisive inserter; cleared under m
    // by L at its next insert). The decide wait sleeps on this word via
    // futex, so it resolves on an O(1) flag with a precise per-lane wake
    // instead of re-running the O(32) peer classification per cv wakeup —
    // that scan (x32 lanes x ~1.2M gathers in CB_strsm) was tens of
    // CPU-minutes of sys time. Instances decided by a peer EXIT or PARK are
    // not stamped; the timed re-check in the decide wait covers those.
    std::atomic<uint32_t> ready[WARP_SIZE] = {};
    // slot_busy[L]==1: e[L] holds an unswept entry. The slot wait sleeps on
    // this word via futex and the sweeper wakes each lane individually —
    // the cv version made all 31 early re-entrants stampede through ctx.m
    // on every sweep notify_all (they loop back to the next gather before
    // the slowest consumer of the previous one has swept). Zero-init = free.
    std::atomic<uint32_t> slot_busy[WARP_SIZE] = {};
};
static WarpGroupCtx* g_warpGroupCtx = nullptr;

// Publish without waking group waiters: used for events that are immediately
// followed by a gather INSERT (which runs the decidability check and notifies
// when warranted). Wake-on-every-publish amplified into a futex storm: 32
// bump-publishes x notify_all(31 waiters) per gather ground CB_strsm's 1.18M
// gathers to a halt (sys-time 10x wall).
static void warp_hist_publish_quiet()
{
    if (!g_warpHistCtx) return;
    const int wg = g_threadCtx.warpGlobal, lane = g_threadCtx.lane;
    std::lock_guard<std::mutex> lk(g_warpHistCtx[wg].m);
    g_warpHistCtx[wg].seq[lane]  = g_laneHist.seq;
    g_warpHistCtx[wg].hash[lane] = g_laneHist.hash;
    g_warpHistCtx[wg].ring[lane] = g_laneHist.ring;
}

static void warp_hist_publish()
{
    if (!g_warpHistCtx) return;
    warp_hist_publish_quiet();
    // Standalone events (region push/pop, thread start) may flip a waiter's
    // classification with no insert following — those must wake waiters.
    if (g_warpGroupCtx) g_warpGroupCtx[g_threadCtx.warpGlobal].cv.notify_all();
}

// Classify peer M relative to caller (my TLS history): returns true if M
// might still join my instance (must wait), false if provably not coming.
static bool peer_may_join_(const WarpHistCtx& h, int M)
{
    const uint64_t mseq = h.seq[M];
    if (h.ring[M] == nullptr) return false;            // exited
    if (h.parked[M]) return false;                      // parked at a barrier
    if (mseq > g_laneHist.seq) return false;            // ahead: passed already
    if (mseq == g_laneHist.seq) {
        // At my event count: in my instance iff identical history. (It has
        // bumped at this collective but may not have inserted its entry yet.)
        return h.hash[M] == g_laneHist.hash;
    }
    if (g_laneHist.seq - mseq >= HIST_RING) return true; // too far to tell: wait
    return g_laneHist.ring[mseq % HIST_RING] == h.hash[M]; // behind on my path?
}

// peer_may_join_ evaluated from lane W's published view instead of the
// calling thread's TLS (h.ring[W] points at W's TLS ring; W is blocked in
// its gather while this is read, so the ring is quiescent). Lets an
// INSERTING lane decide whether instance-hash H became decidable for its
// waiters, so it can notify only then (see the insert step below).
static bool peer_may_join_view_(const WarpHistCtx& h, int W, int M)
{
    const uint64_t wseq = h.seq[W], mseq = h.seq[M];
    if (h.ring[M] == nullptr) return false;            // exited
    if (h.parked[M]) return false;                      // parked at a barrier
    if (mseq > wseq) return false;                      // ahead: passed already
    if (mseq == wseq) return h.hash[M] == h.hash[W];
    if (wseq - mseq >= HIST_RING) return true;          // too far to tell
    return h.ring[W][mseq % HIST_RING] == h.hash[M];    // behind on W's path?
}

// Diagnostic twin of peer_may_join_: returns the reason a peer is excluded
// (0=may join, 1=exited, 2=parked, 3=ahead, 4=same-seq hash mismatch,
// 5=ring mismatch). Used by the SLIFTER_GATHER_TRACE reporter below.
static int peer_may_join_dbg_(const WarpHistCtx& h, int M)
{
    const uint64_t mseq = h.seq[M];
    if (h.ring[M] == nullptr) return 1;                 // exited
    if (h.parked[M]) return 2;                          // parked
    if (mseq > g_laneHist.seq) return 3;                // ahead
    if (mseq == g_laneHist.seq)
        return (h.hash[M] == g_laneHist.hash) ? 0 : 4;  // same-seq hash mismatch
    if (g_laneHist.seq - mseq >= HIST_RING) return 0;
    return (g_laneHist.ring[mseq % HIST_RING] == h.hash[M]) ? 0 : 5; // ring mismatch
}

// Gather the caller's group: blocks until membership is decided, returns the
// member bitmask; vals/preds of members are copied into out arrays.
// `site` identifies the static collective so distinct collectives chain
// distinct hashes. It is the SASS instruction address, passed down from the
// lifted IR as an explicit argument — NOT the binary call site
// (__builtin_return_address): clang -O2 tail-duplicates lifted blocks, so
// one logical collective can have several binary call sites, and lanes
// reaching it through different CFG paths would chain different hashes and
// split the warp into disjoint groups (CB_strsm trsm: every solve-loop
// SHFL.IDX returned the caller's own value).
static uint32_t warp_group_gather(uint32_t site, uint32_t payload, bool pred,
                                  uint32_t vals_out[WARP_SIZE],
                                  bool preds_out[WARP_SIZE],
                                  bool full_warp = false, int full_live = 0)
{
    // full_warp: a warp-synchronous SHFL with a full-warp membership mask
    // (0xffffffff). On HW such a shuffle implicitly RECONVERGES all active lanes,
    // but the divergence-history matcher (which fragments lanes that took
    // different past paths — e.g. FA consumer warps after warp-spec, whose
    // per-lane seq counters are offset and never reconverge because BSSY/BSYNC
    // is DCE'd) would return singleton groups => no reduction. In full_warp mode
    // the group is decided by arrived==full_live (all live warp lanes present)
    // and membership is all arrived lanes, regardless of history hash. Safe: a
    // genuinely-divergent shuffle reaching <full_live lanes never sets full_warp
    // at a site only some lanes execute — but the lifter passes mask=0xffffffff
    // unconditionally, so this assumes every full-mask SHFL is truly warp-wide.
    // Lockstep mode: a lane inside a collective does no token-gated shared
    // accesses — leave the turn rotation so peers can keep stepping.
    TurnParkGuard turn_park_;
    const int wg = g_threadCtx.warpGlobal, lane = g_threadCtx.lane;
    WarpGroupCtx&  ctx = g_warpGroupCtx[wg];
    WarpHistCtx&   h   = g_warpHistCtx[wg];

    // The collective itself is a history event. Bumping (seq, hash) BEFORE
    // gathering is what lets peers distinguish "same path, not arrived yet"
    // (their ring at my seq-1 holds my previous hash -> must wait) from
    // "already passed this instance" — without it, all lanes of an
    // event-free path sit at the same seq and the first arriver decides a
    // singleton group (observed: tile shuffles returning own values).
    g_laneHist.hash = (g_laneHist.hash ^ (uint64_t(site) | 0x300000000ull)) * 0x100000001b3ull;
    g_laneHist.seq++;
    g_laneHist.ring[g_laneHist.seq % HIST_RING] = g_laneHist.hash;
    // Quiet publish: the insert below runs the decidability check and wakes
    // waiters only when an instance actually resolved.
    warp_hist_publish_quiet();
    const uint64_t H = g_laneHist.hash;

    if (getenv("SLIFTER_SEQ_DEBUG") && g_threadCtx.warpInBlock==4 && lane<4) {
        static std::atomic<int> nq{0};
        if (nq.fetch_add(1) < 32)
            std::fprintf(stderr, "[seq] cta=%d warp=%d lane=%d site=0x%x seq=%llu hash=%llx\n",
                         g_threadCtx.cta, g_threadCtx.warpInBlock, lane, site,
                         (unsigned long long)g_laneHist.seq, (unsigned long long)H);
    }

    if (const uint64_t n_ = spin_hotcall("warp_group_gather"))
        std::fprintf(stderr, "HOTCALL site=warp_group_gather tid=%d cta=%d lane=%d "
                     "callsite=0x%x count=%llu\n", g_threadCtx.tid, g_threadCtx.cta,
                     lane, site, (unsigned long long)n_);

    std::unique_lock<std::mutex> lock(ctx.m);
    // My previous instance's slot must have been swept before reuse (per-
    // lane futex; the sweeper wakes exactly this lane).
    {
        const auto st0 = std::chrono::steady_clock::now();
        bool sreported = false;
        while (ctx.e[lane].valid) {
            lock.unlock();
            if (warp_fiber_active()) g_warp_yield();   // fiber: yield, don't block the CTA's OS thread
            else futex_wait_u32(&ctx.slot_busy[lane], 1, 50);
            lock.lock();
            if (spin_debug_on() && !sreported &&
                std::chrono::steady_clock::now() - st0 > std::chrono::seconds(5)) {
                sreported = true;
                std::fprintf(stderr, "SPIN site=gather_slot tid=%d cta=%d lane=%d wg=%d "
                             "callsite=0x%x stale_hash=%llx consumed=%d\n",
                             g_threadCtx.tid, g_threadCtx.cta, lane, wg, site,
                             (unsigned long long)ctx.e[lane].hash, (int)ctx.e[lane].consumed);
            }
        }
    }
    ctx.e[lane] = GroupEntry{true, false, pred, payload, H};
    ctx.ready[lane].store(0, std::memory_order_relaxed);
    ctx.slot_busy[lane].store(1, std::memory_order_relaxed);
    // Decide-and-stamp: if this insert made some present instance decidable,
    // mark all its arrived members ready and futex-wake exactly them.
    // Waiters resolve on the O(1) flag instead of re-running the peer scan
    // per wakeup. Every insert-decidable transition is accompanied by an
    // insert (the bump-publish immediately precedes it); exit/park-driven
    // decidability is covered by the timed re-check in the decide wait
    // below. Unconditional cv notify_all here (32 per gather, each waking
    // up to 31 waiters that re-scan and re-sleep) was the futex storm that
    // made CB_strsm's ~1.2M gathers take hours.
    uint32_t stamp_mask = 0;
    {
        std::lock_guard<std::mutex> hk(h.m);
        for (int W = 0; W < WARP_SIZE; ++W) {
            if (!ctx.e[W].valid || ctx.e[W].consumed) continue;
            if (ctx.ready[W].load(std::memory_order_relaxed))
                continue;            // instance already decided + stamped
            const uint64_t HW = ctx.e[W].hash;
            bool first = true;       // evaluate each distinct instance hash once
            for (int V = 0; V < W; ++V)
                if (ctx.e[V].valid && !ctx.e[V].consumed && ctx.e[V].hash == HW) {
                    first = false; break;
                }
            if (!first) continue;
            bool decidable = true;
            for (int M = 0; M < WARP_SIZE; ++M) {
                if (ctx.e[M].valid && ctx.e[M].hash == HW) continue;  // arrived
                if (peer_may_join_view_(h, W, M)) { decidable = false; break; }
            }
            if (!decidable) continue;
            for (int M = 0; M < WARP_SIZE; ++M)
                if (ctx.e[M].valid && ctx.e[M].hash == HW) {
                    ctx.ready[M].store(1, std::memory_order_release);
                    stamp_mask |= (1u << M);
                }
        }
    }
    // Wake after dropping h.m; ctx.m is still held, but FUTEX_WAKE does not
    // touch user-space state, and woken waiters re-read the flag without
    // needing ctx.m. No wake for self: the flag check below never sleeps.
    for (int M = 0; M < WARP_SIZE; ++M)
        if ((stamp_mask >> M) & 1 && M != lane)
            futex_wake_u32(&ctx.ready[M], 1);

    auto decided = [&]{
        // full_warp: decide when every lane that can still participate has
        // deposited. `full_live` is the physical tail-warp width, but lanes
        // that EXITed or are parked at a block barrier are not in the current
        // active set. Requiring the original width deadlocks a SHFL in one
        // divergent arm while sibling lanes wait at the following BAR.SYNC
        // (HeCBench filter_shared).
        if (full_warp) {
            int cnt = 0;
            for (int M = 0; M < WARP_SIZE; ++M) if (ctx.e[M].valid) ++cnt;
            int can_arrive = 0;
            {
                std::lock_guard<std::mutex> hk(h.m);
                for (int M = 0; M < full_live; ++M)
                    if (h.ring[M] != nullptr && !h.parked[M]) ++can_arrive;
            }
            return cnt >= can_arrive;
        }
        // Fast path: the decisive inserter stamped my instance (O(1)).
        // True for the decisive inserter itself (its own sweep above), so
        // the last arriver never sleeps.
        if (ctx.ready[lane].load(std::memory_order_acquire)) return true;
        // Backstop scan: instances decided by peer exit/park are never
        // stamped — evaluated on entry and on each timed re-check.
        std::lock_guard<std::mutex> hk(h.m);
        for (int M = 0; M < WARP_SIZE; ++M) {
            if (M == lane) continue;
            // Arrived in my instance (consumed or not): hash is instance-
            // unique because every gather chains it.
            if (ctx.e[M].valid && ctx.e[M].hash == H) continue;
            if (peer_may_join_(h, M)) return false;    // still possible: wait
        }
        return true;
    };
    auto report_gather_decide_ = [&]{
        uint32_t arrived_m = 0, mayjoin_m = 0, parked_m = 0, exited_m = 0;
        char seqs[224]; int sp = 0; seqs[0] = '\0';
        {
            std::lock_guard<std::mutex> hk(h.m);
            for (int M = 0; M < WARP_SIZE; ++M) {
                if (ctx.e[M].valid && ctx.e[M].hash == H) arrived_m |= 1u << M;
                if (h.parked[M]) parked_m |= 1u << M;
                if (h.ring[M] == nullptr) exited_m |= 1u << M;
                if (M != lane && peer_may_join_(h, M)) {
                    mayjoin_m |= 1u << M;
                    if (sp < (int)sizeof(seqs) - 24)
                        sp += snprintf(seqs + sp, sizeof(seqs) - sp, " M%d:seq=%llu",
                                       M, (unsigned long long)h.seq[M]);
                }
            }
        }
        std::fprintf(stderr, "SPIN site=gather_decide tid=%d cta=%d lane=%d wg=%d "
                     "callsite=0x%x myseq=%llu arrived=%08x mayjoin=%08x parked=%08x "
                     "exited=%08x%s\n", g_threadCtx.tid, g_threadCtx.cta, lane, wg,
                     site, (unsigned long long)g_laneHist.seq, arrived_m, mayjoin_m,
                     parked_m, exited_m, seqs);
    };
    // Decide wait: sleep on my own ready word (precise wake by the decisive
    // inserter); 50ms timed re-check as the backstop for exit/park-decided
    // instances, which are never stamped. The flag/futex race is handled by
    // FUTEX_WAIT's atomic expect-check; entries are re-read under ctx.m
    // after the wait, so the flag itself carries no data dependency.
    {
        const auto dt0 = std::chrono::steady_clock::now();
        bool dreported = false;
        while (!decided()) {
            lock.unlock();
            if (warp_fiber_active()) g_warp_yield();   // fiber: yield so peer lanes run + arrive
            else futex_wait_u32(&ctx.ready[lane], 0, 50);
            lock.lock();
            if (spin_debug_on() && !dreported &&
                std::chrono::steady_clock::now() - dt0 > std::chrono::seconds(5)) {
                dreported = true;
                report_gather_decide_();
            }
        }
    }

    uint32_t mask = 0;
    for (int M = 0; M < WARP_SIZE; ++M) {
        // full_warp: membership = all arrived lanes (reconvergence), ignoring hash.
        if (ctx.e[M].valid && (full_warp || ctx.e[M].hash == H)) {
            mask |= (1u << M);
            vals_out[M]  = ctx.e[M].val;
            preds_out[M] = ctx.e[M].pred;
        }
    }
    // SLIFTER_GATHER_TRACE=N: report the first N gathers that resolve to a
    // non-full group, with each missing peer's exclusion reason. Non-full
    // groups are legal under real divergence/exit, but a kernel that should
    // be converged showing splits here means the history hashes diverged —
    // exactly how the CB_strsm own-value-shuffle bug was found.
    {
        static std::atomic<long> gn_{0};
        static long glim_ = []{ const char* e = std::getenv("SLIFTER_GATHER_TRACE");
                                return e ? std::atol(e) : 0; }();
        if (glim_ && gn_.fetch_add(1) < glim_ && mask != 0xffffffffu) {
            char buf[512]; int bp = 0;
            {
                std::lock_guard<std::mutex> hk(h.m);
                for (int M = 0; M < WARP_SIZE && bp < (int)sizeof(buf) - 48; ++M) {
                    if ((mask >> M) & 1) continue;
                    bp += snprintf(buf + bp, sizeof(buf) - bp, " M%d:r%d,seq=%llu,h=%llx",
                                   M, peer_may_join_dbg_(h, M),
                                   (unsigned long long)h.seq[M],
                                   (unsigned long long)(h.hash[M] & 0xffff));
                }
            }
            std::fprintf(stderr, "GATHERTRACE wg=%d lane=%d site=0x%x myseq=%llu myh=%llx "
                         "mask=%08x missing:%s\n", wg, lane, site,
                         (unsigned long long)g_laneHist.seq,
                         (unsigned long long)(H & 0xffff), mask, buf);
        }
    }
    ctx.e[lane].consumed = true;
    // Last member of this group sweeps the group's slots for reuse and
    // futex-wakes each slot-waiter individually. Only the SWEEP changes
    // observable state on the way out: slot-waiters need exactly that
    // event, and a member's consume cannot affect any decide wait
    // (consumed entries still count as arrived; classifications read
    // publish state, not entries).
    bool all_consumed = true;
    for (int M = 0; M < WARP_SIZE; ++M)
        if ((mask >> M) & 1 && !ctx.e[M].consumed) { all_consumed = false; break; }
    if (all_consumed)
        for (int M = 0; M < WARP_SIZE; ++M)
            if ((mask >> M) & 1) {
                ctx.e[M] = GroupEntry{};
                ctx.slot_busy[M].store(0, std::memory_order_release);
            }
    lock.unlock();
    if (all_consumed)
        for (int M = 0; M < WARP_SIZE; ++M)
            if ((mask >> M) & 1 && M != lane)
                futex_wake_u32(&ctx.slot_busy[M], 1);
    return mask;
}

// Explicit SASS WARPSYNC.ALL. This is stronger than a history event: all
// currently live/unparked lanes rendezvous, then their divergence histories
// are reset to the same epoch so a subsequent history-matched collective sees
// one reconverged warp. Each launch uses fresh std::threads, so this TLS counter
// starts from zero for every logical CUDA lane.
extern "C" void warp_sync_all(uint32_t site)
{
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp =
        std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return;

    uint32_t vals[WARP_SIZE] = {};
    bool preds[WARP_SIZE] = {};
    (void)warp_group_gather(
        site, 0, true, vals, preds, true, lanesInWarp);

    static thread_local uint64_t sync_epoch = 0;
    const uint64_t epoch =
        (++sync_epoch << 32) | static_cast<uint64_t>(site);
    warp_hist_resync_after_barrier(epoch);
}

// Snapshot of const_mem after argument marshalling
static std::array<uint8_t, sizeof(const_mem)> g_constMemTemplate{};

// For cuBLAS/cuDNN drivers: write struct params to main thread's const_mem.
// Call BEFORE launchKernel() — it snapshots const_mem into the template.
template<typename T>
inline void writeStructField(int offset, T val) {
    std::memcpy(&const_mem[0][offset], &val, sizeof(T));
}

// -----------------------------------------------------------------------------
// EXIT-aware warp-collective completion
//
// The dynamic membership of a warp collective is the set of lanes still
// executing (divergent guard-then-collective patterns: non-participating
// lanes run to EXIT). An instance completes when every live lane of the
// warp has arrived; lane exits (which decrement ctx.live and notify) can
// also complete an instance, in which case the first waiting lane to wake
// runs the finalize step. Mirrors the live-thread barrier semantics in
// BlockSyncCtx.
// -----------------------------------------------------------------------------
template <typename Ctx, typename Finalize>
inline void warpCollectiveComplete(Ctx& ctx, std::unique_lock<std::mutex>& lock,
                                   int myGen, Finalize&& fin)
{
    // Serialized-warp fiber path: one OS thread, so YIELD until every lane has
    // arrived (a condvar wait would freeze the whole CTA). The lane that observes
    // all-arrived runs fin() once and bumps the generation; earlier arrivers see
    // the generation change and return. Covers all matrix collectives
    // (STSM/HMMA/LDSM/DMMA/IMMA) that call this.
    if (warp_fiber_active()) {
        if (ctx.arrived >= ctx.live) { fin(); ctx.arrived = 0; ++ctx.generation; return; }
        while (ctx.generation == myGen && ctx.arrived < ctx.live) {
            lock.unlock(); g_warp_yield(); lock.lock();
        }
        if (ctx.generation == myGen && ctx.arrived >= ctx.live) {
            fin(); ctx.arrived = 0; ++ctx.generation;
        }
        return;
    }
    // Lockstep mode: leave the turn rotation while blocked here (turn mutex
    // is a leaf — taking it under the collective's lock cannot cycle).
    TurnParkGuard turn_park_;
    if (ctx.arrived >= ctx.live) {
        fin();
        ctx.arrived = 0;
        ++ctx.generation;
        lock.unlock();
        ctx.cv.notify_all();
        return;
    }
    spin_diag_wait(ctx.cv, lock, [&] {
        if (ctx.generation != myGen)
            return true;
        if (ctx.arrived > 0 && ctx.arrived >= ctx.live) {
            fin();
            ctx.arrived = 0;
            ++ctx.generation;
            ctx.cv.notify_all();
            return true;
        }
        return false;
    }, [&]{
        std::fprintf(stderr, "SPIN site=warp_collective tid=%d cta=%d lane=%d "
                     "arrived=%d live=%d gen=%d mygen=%d\n",
                     g_threadCtx.tid, g_threadCtx.cta, g_threadCtx.lane,
                     ctx.arrived, ctx.live, ctx.generation, myGen);
    });
}

// -----------------------------------------------------------------------------
// Shuffle helpers
// -----------------------------------------------------------------------------

// Interpret clamp as used by the lifted IR:
//  - clamp == 0 comes from SASS "RZ" and means "full width" (whole warp).
//  - otherwise: width = min((clamp & 0x1f) + 1, lanesInWarp).
// HW reads maxLane from c[4:0] only (segmask lives at c[12:8]), so runtime
// c values like -1 (CUB partial-tile clamp for lanes with no valid items)
// mean maxLane=31 on the GPU — a plain clamp+1 would give width 0 and
// SIGFPE on the lane/width divide (mtf-cuda find_if reduce).
inline int shfl_effective_width(int clamp, int lanesInWarp) {
    if (clamp == 0)
        return lanesInWarp;
    int width = (clamp & 0x1f) + 1;
    if (width > lanesInWarp)
        width = lanesInWarp;
    return width;
}

template<typename T>
T warp_shfl_down(uint32_t mask,
       T       val,
       int32_t offset,
       int32_t clamp,
       uint32_t site)
{
    (void)mask;
    static_assert(sizeof(T) == 4, "warp_shfl_down only supports 32-bit types");
    const int lane        = g_threadCtx.lane;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return val;
    const int width = shfl_effective_width(clamp, lanesInWarp);

    uint32_t raw; std::memcpy(&raw, &val, 4);
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, raw, true, vals, preds);

    int srcLane = lane + offset;
    // Segment semantics within the warp; out-of-window or non-member source
    // keeps the caller's own value (CUDA: undefined / self for inactive src).
    const int seg = (lane / width) * width;
    uint32_t outRaw = raw;
    if (srcLane >= seg && srcLane < seg + width && srcLane < WARP_SIZE
        && ((members >> srcLane) & 1))
        outRaw = vals[srcLane];
    T out; std::memcpy(&out, &outRaw, sizeof(T));
    return out;
}

template<typename T>
T warp_shfl_up(uint32_t mask,
       T       val,
       int32_t offset,
       int32_t clamp,
       uint32_t site)
{
    (void)mask;
    static_assert(sizeof(T) == 4, "warp_shfl_up only supports 32-bit types");
    const int lane        = g_threadCtx.lane;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return val;
    const int width = shfl_effective_width(clamp, lanesInWarp);

    uint32_t raw; std::memcpy(&raw, &val, 4);
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, raw, true, vals, preds);

    int srcLane = lane - offset;
    // Segment semantics within the warp; out-of-window or non-member source
    // keeps the caller's own value (CUDA: undefined / self for inactive src).
    const int seg = (lane / width) * width;
    uint32_t outRaw = raw;
    if (srcLane >= seg && srcLane < seg + width && srcLane < WARP_SIZE
        && ((members >> srcLane) & 1))
        outRaw = vals[srcLane];
    T out; std::memcpy(&out, &outRaw, sizeof(T));
    return out;
}

// C wrappers matching the lifted IR declarations

// -----------------------------------------------------------------------------
// TMA (Tensor Memory Accelerator) — x86 model
//
// On real sm_90a a UTMALDG bulk-copies a tile described by a host-encoded
// CUtensorMap from global into shared memory (with an optional 128B swizzle).
// The CUtensorMap is opaque, so on x86 the generated host driver materializes
// its OWN descriptor of this exact shape at the address the kernel's descriptor
// load resolves to, and lift_utmaldg lowers each UTMALDG to one call here. This
// performs the genuine coord -> (swizzled) shared-memory tile copy that
// populates the operands the downstream HGMMA reads back out of shared memory.
// -----------------------------------------------------------------------------
struct SlifterTmaDesc {
    const void* global_base;   // qword 0: tensor base in (host) global memory
    int32_t     rows;          // tile rows  (M for A, N for B)
    int32_t     cols_k;        // total tile K extent (all K-blocks)
    int32_t     elem_bytes;    // 2 for fp16/bf16
    int32_t     swizzle;       // 0=none(K_INTER), 1=SW32, 2=SW64, 3=SW128
    int32_t     ld;            // global leading dim in elements (row stride); 0 => cols_k
    // K>8 tiles are issued as multiple UTMALDG, one per 8-wide K-block, each to
    // a different smem offset (K-INTER: block j at tile_base + j*rows*8*2 bytes).
    // The host records this tile's base smem offset so utma_load() can recover
    // the K-block index for a per-call smem_off and copy only that block's
    // global K-columns (otherwise later blocks would clobber earlier ones).
    int32_t     tile_base_off; // smem byte offset of this tile's K-block 0
    // Valid global rows of THIS (head,batch) plane (= seqlen). The K-loop's last
    // n_block is a PARTIAL tile (rows beyond seqlen) that FA masks in softmax; the
    // global read of those rows is out of the host buffer, so utma_load clamps
    // (m_base+m) < max_rows and zero-fills the rest. 0 = no clamp (legacy).
    int32_t     max_rows;
    // Per-mode global strides (in elements) for the OUTER TMA coordinates c2/c3.
    // A real CUtensorMap encodes one stride per tensor dimension; the 2D
    // (rows x ld) model above only captured modes c0 (DIM, stride 1) and c1
    // (SEQ, stride ld). For a multi-(head,batch) tensor the kernel passes the
    // head as c2 and batch as c3, so utma_load/store must step the global base
    // by c2*stride_c2 + c3*stride_c3. e.g. [B,SQ,H,D] row-major => stride_c2
    // (head) = D, stride_c3 (batch) = SQ*H*D. 0 = single-plane (legacy: outer
    // offset 0, matching the only-ever-verified c2=c3=0 path).
    int32_t     stride_c2;     // global stride (elems) per unit of coord c2 (head)
    int32_t     stride_c3;     // global stride (elems) per unit of coord c3 (batch)
    // FA3 FP8 only: V is row-major in gmem and transposed in-kernel (LDSM.T /
    // STSM) before the PV GEMM. cute loads V into smem with SmemLayoutVt
    // (TmaMajorV=K over (kHeadDimV, kBlockN) with Step<_1,_2>) — a DIFFERENT
    // physical swizzle from the K-major Q/K SW128 layout: seqlen is the
    // contiguous (stride-1) dim, hdim is atom-strided. The in-kernel LDSM.T then
    // reads exactly this layout. If utma_load writes V with the Q/K K-major
    // swizzle (hdim contiguous) instead, the LDSM.T reads the wrong/zero bytes
    // for half the V tile. Set vt_load=1 for the V descriptor so utma_load uses
    // the SmemLayoutVt closed form. 0 = normal SW128 K-major (Q/K, all fp16).
    int32_t     vt_load;
    // FA3 BACKWARD dKV store only: the dK and dV epilogue STORE accumulators
    // interleave in ONE smem segment (dK @ region base 0, dV @ region base
    // 0x8000), each a key-major SW128 tile (SmemLayoutdKV = tile_to_shape(
    // ss_smem_selector<K,...>(kBlockN, kHeadDim/AtomLayoutKdKV), (kBlockN,
    // kHeadDim))). The headdim ATOM index must therefore be computed RELATIVE TO
    // each tensor's smem region base — NOT absolutely as the fwd O store does
    // (whose two O atoms live at smem 0 / 0x4000). Absolute indexing sent dV's
    // 0x8000/0xc000 atoms to global hd 128/192 (out of the 0-127 range). The
    // WITHIN-ATOM (key,hd)->smem map is the SAME as the fwd O store (the dKV R2S
    // copy is the same STSM_N atom, dKV_swapAB=false): plain SW128 + the mr/corr
    // bit swizzle. Value-verified vs CPU dV: fwd mr/corr map -> dV atom-0 corr
    // 1.0000; plain SW128 -> 0.63 in the validation sweep.
    // The driver sets store_dkv = (region_base_bytes + 1) for the dK/dV store
    // descriptors (0 = fwd O-store, unchanged path; nonzero => bwd dKV store,
    // region_base = store_dkv - 1).
    int32_t     store_dkv;
    // SW-path swizzle BIT-WIDTH B for the in-smem tile layout (the WRITE-side
    // twin of the lifter's _sw128_phys_off swiz_B read). B=3 -> SW128 ATOM 64,
    // B=2 -> SW64 ATOM 32, B=1 -> SW32 ATOM 16 (ATOM = 8<<B). 0 (the zero-init
    // default) is treated as 3 == SW128, so every existing driver — which leaves
    // this field at 0 via `SlifterTmaDesc d{}` — gets the BYTE-IDENTICAL SW128
    // layout. hd96's K is SW32 (swiz_b=1), Q/dO/V are SW64 (swiz_b=2); their
    // narrower atoms shrink the TMA tile so it no longer aliases the LSE tile.
    // Only consulted on the swizzle==3 SW path (swizzle 0/4 are K-INTER/linear).
    int32_t     swiz_b;
};

// SmemLayoutVt(hdim, seqlen) for FA3 FP8 (e4m3): the physical ELEMENT offset of
// V in smem after the TMA load (before the in-kernel transpose). Probed exact
// vs cute (0/28672 mismatches):
//   Sw<1,4,3> o ((_8,_16),(_32,_7)):((_32,_256),(_1,_4096))
//   idx = (h%8)*32 + (h/8)*256 + (s%32) + (s/32)*4096 ; ^ (((idx>>7)&1)<<4)
// hdim is split (8,16):(32,256), seqlen split (32, kBlockN/32):(1,4096), with the
// 1-bit Swizzle<1,4,3>. (NOT the 3-bit Swizzle<3,4,3> the K-major Q/K tiles use.)
static inline int slifter_vt_load_elem(int hdim, int seq) {
    int idx = (hdim & 7) * 32 + (hdim >> 3) * 256 + (seq & 31) + (seq >> 5) * 4096;
    return idx ^ (((idx >> 7) & 1) << 4);
}

// K-major GMMA smem physical offset (in elements) of logical (m,k).
//   no swizzle (K_INTER):  off = m*8 + (k&7) + (k>>3)*rows*8
// This is cute's Layout_K_INTER_Atom tiled over (rows, cols_k); for a single
// K-atom (cols_k<=8) it reduces to the linear m*cols_k + k order.
static inline int slifter_kinter_off(int m, int k, int rows) {
    return m * 8 + (k & 7) + (k >> 3) * rows * 8;
}

// 128B-swizzle applied to a linear ELEMENT index within the smem tile.
// cute's Layout_K_SW128_Atom is ComposedLayout<Swizzle<3,4,3>, ...,
// Shape<8,1024>:Stride<1024,1>> in BITS, upcast to elements. Probed exactly
// against cute: for a K-major tile of K_total columns the
// physical element offset of logical (m,k) is the row-major linear index
// (m*K_total + k) put through Swizzle<B=3,M=4,S=3> ON THE ELEMENT INDEX:
//   yyy = (e >> (M+S)) & ((1<<B)-1) = (e >> 7) & 7 ;  e ^= yyy << M  (<<4)
// This is NOT "K-INTER then byte-swizzle" — the SW128 base layout is plain
// row-major, distinct from K_INTER's m*8 + ... block layout.
static inline int slifter_sw128_elem(int e) {
    int yyy = (e >> 7) & 0x7;          // (e >> (M+S)) & 7  ; M+S=4+3=7
    return e ^ (yyy << 4);             // ^ (yyy << M) ; M=4
}

// GENERAL GMMA swizzle on a linear ELEMENT index, parametrized by the swizzle
// bit-width B: cute's Layout_K_SW{128,64,32}_Atom is Swizzle<B,4,3> over the
// element index (B=3 -> SW128 ATOM 64 elems, B=2 -> SW64 ATOM 32, B=1 -> SW32
// ATOM 16; ATOM = 8<<B). The XOR field shifts by M=4, the period is M+S=7:
//   yyy = (e >> 7) & ((1<<B)-1) ;  e ^= yyy << 4
// B=3 reproduces slifter_sw128_elem() BYTE-FOR-BYTE (mask 7). This is the
// WRITE-side twin of the lifter's _sw128_phys_off(...,swiz_B) read-side decode
// (lift/lifter_cpu.py): both must lay/read the tile with the SAME ATOM=8<<B and
// mask (1<<B)-1 so the TMA-written tile and the WGMMA SS-read agree. hd96's
// K is SW32 (B1), Q/dO/V are SW64 (B2) — mixed swizzle the old SW128-only path
// could not represent (it aliased + clobbered the LSE tile, dK/dV/dQ blew up).
static inline int slifter_swizzle_elem(int e, int B) {
    if (B == 3) return slifter_sw128_elem(e);   // SW128 fast path, byte-identical
    int yyy = (e >> 7) & ((1 << B) - 1);
    return e ^ (yyy << 4);
}

// dKV epilogue STORE within-atom element map for a SW64 (B=2, ATOM=32) dKV tile.
// The R2S copy is make_tiled_copy_C(SM90_U32x4_STSM_N, TiledMmadKV); the STSM_N writes
// the dKV MMA C-fragment into SmemLayoutdKV (cute Sw<2,4,3>) with a (key,hd)->smem element
// map that the SW128 mr/corr store (B=3) does NOT reproduce. The map is AFFINE over the
// bits of (key 0..127, kk = hd within the 32-wide atom 0..31). Decoded noise-free via
// source-id injection into stsm_m88_warp (SLIFTER_DKV_INJECT) joined with the
// cute MMA-C fragment map: within-atom
// 4096/4096 exact, dV b0h0 corr 1.0. Returns the within-atom smem element offset (the store
// reads from the per-atom smem base ssrc, so atom/headdim is already folded into ssrc).
// b1,b2 (key bits 1,2) carry the Sw<2,4,3> swizzle into kk bits 3,4. hd96 dKV store ONLY.
static inline int slifter_dkv_stsm_sw64(int key, int kk) {
    const int b0 = key & 1, b1 = (key >> 1) & 1, b2 = (key >> 2) & 1, b3 = (key >> 3) & 1;
    const int b4 = (key >> 4) & 1, b5 = (key >> 5) & 1, b6 = (key >> 6) & 1;
    const int h0 = kk & 1, h1 = (kk >> 1) & 1, h2 = (kk >> 2) & 1, h3 = (kk >> 3) & 1, h4 = (kk >> 4) & 1;
    return (h0 << 0) | (h1 << 1) | (h2 << 2) | ((h3 ^ b1) << 3) | ((h4 ^ b2) << 4)
         | (b0 << 5) | (b1 << 6) | (b2 << 7) | (b3 << 8) | (b4 << 9) | (b5 << 10) | (b6 << 11);
}

// Physical element offset of logical (m,k) for an SW128 K-major tile of
// K_total columns: row-major linear index then the element-index swizzle.
static inline int slifter_sw128_off(int m, int k, int K_total) {
    return slifter_sw128_elem(m * K_total + k);
}

// Forward decl: utma_load reports its delivered tile bytes to the expect-tx
// byte-accounting FIFO (defined below). No-op when nothing is pending.
extern "C" void slifter_mbar_deliver_tx(int64_t bytes);

// DEBUG (gated SLIFTER_LSE_TRACE): unified data-lifetime tracker for the FA3 bwd
// LSE/dPsum tile. Declared here so utma_load (above the dump defs) can record TMA
// tiles that overlap the window. Definitions follow; the constants/state live in
// this block so both utma_load and the dump hooks share one seq counter.
static std::atomic<long> g_lse_seq{0};
static constexpr uint32_t LSE_TRACE_LO = 0x2e000u;
static constexpr uint32_t LSE_TRACE_HI = 0x2e6c0u;   // exclusive upper bound
static constexpr uint32_t LSE_TRACE_SLOTS = (LSE_TRACE_HI - LSE_TRACE_LO) / 8u;
struct LseSlotInfo { long seq; float val; char kind; };  // kind: 'u'=ublkcp 's'=sts 't'=tma
static LseSlotInfo g_lse_last_write[LSE_TRACE_SLOTS];     // zero-init -> seq=0
static inline bool lse_trace_on() {
    static const int e = std::getenv("SLIFTER_LSE_TRACE") ? 1 : 0; return e;
}
static inline void lse_trace_record_write(uint32_t off, uint32_t bytes,
                                          const float* src, char kind, long seq) {
    // mark every 8-byte slot the write covers within the tracked window; record
    // the LIVE destination smem value (post-write) so a later clobber is detectable.
    (void)src;
    uint32_t end = off + bytes;
    uint32_t a = off < LSE_TRACE_LO ? LSE_TRACE_LO : off;
    uint32_t b = end > LSE_TRACE_HI ? LSE_TRACE_HI : end;
    for (uint32_t p = a; p < b; p += 8) {
        uint32_t s = (p - LSE_TRACE_LO) / 8u;
        float v = 0.f;
        std::memcpy(&v, reinterpret_cast<const uint8_t*>(shared_mem) + p, sizeof(float));
        g_lse_last_write[s] = { seq, v, kind };
    }
}

extern "C" void utma_load(uint64_t desc_addr, uint8_t* smem_base,
                          int32_t smem_off,
                          int32_t c0, int32_t c1, int32_t c2, int32_t c3)
{
    // TMA bulk-tensor copy is a WARP-UNIFORM single-issue operation.  The tile
    // coordinates {c0..c3} live in uniform registers (UR*) written by the
    // ELECTED lane (cute issues SM90_TMA_LOAD from the lowest active lane via
    // `elect.sync`); on hardware the warp's uniform register file makes that one
    // value visible to all lanes and the copy lands once.  Our x86 model gives
    // each thread its OWN register file, so only the elected lane (lane 0) holds
    // the valid coords — the other lanes carry stale garbage.  Issue the copy
    // once per warp from lane 0 (the elected/issuing lane), matching the HW.
    // General + toggle-free: applies to every UTMALDG regardless of kernel.
    if (g_threadCtx.lane != 0) return;
    const SlifterTmaDesc* d =
        reinterpret_cast<const SlifterTmaDesc*>(desc_addr);
    if (!d || !d->global_base) return;
    const int rows = d->rows, K = d->cols_k, eb = d->elem_bytes;
    const int ld = d->ld ? d->ld : K;

    // 4D tensor coordinates {c0,c1,c2,c3} from the cute SM90_TMA_LOAD ABI window
    // (UR10..UR13), recovered by TmaCoordRecover. For a K-major Q/K/V tile:
    //   c0 = mode-0 (DIM/K) base offset -> k_base (the K-block step; 0 for the
    //        single-K-atom tile, varies for multi-K tiles)
    //   c1 = mode-1 (SEQ/M) base offset -> m_base (the seq-tile ROW; this is the
    //        coordinate that steps per loop iteration — THE thing that was lost)
    //   c2,c3 = outer modes (head/batch) folded into the global base below.
    // A 2D single-tile UTMALDG (or any op with no recovered coords) gets all
    // coords 0 — identical to the prior coord-less behaviour.
    int k_base = c0;
    int m_base = c1;
    // Outer-mode (head/batch) offset folded into the global base, in elements.
    // Each TMA tensor mode has its OWN global stride (a CUtensorMap encodes one
    // stride per dim); c2=head steps by stride_c2, c3=batch by stride_c3. The
    // old (c2+c3)*ld*rows guess was only ever exercised at c2=c3=0 (single
    // plane), so it silently read the wrong plane for any other (head,batch).
    long long outer_elems =
        (long long)c2 * (long long)d->stride_c2 +
        (long long)c3 * (long long)d->stride_c3;

    const uint8_t* gbase = reinterpret_cast<const uint8_t*>(d->global_base)
                           + (size_t)outer_elems * eb;
    uint8_t* sbase = smem_base + d->tile_base_off;

    // Pure-logging diagnostic (no behavior change): dump the recovered coords +
    // computed source base so we can see WHICH coord is garbage when the TMA
    // source address goes wild. Enabled only by SLIFTER_TMA_DEBUG.
    if (getenv("SLIFTER_TMA_DEBUG")) {
        static std::atomic<int> n_dbg{0};
        if (n_dbg.fetch_add(1) < 400) {
            bool bad = (c2 < 0 || c2 > 1024 || c3 < 0 || c3 > 1024 ||
                        m_base < 0 || m_base > 65536);
            fprintf(stderr,
                "[utma_load] cta=%d tid=%d wg=%d desc=%p sw=%d rows=%d K=%d ld=%d "
                "c0=%d c1=%d c2=%d c3=%d -> m_base=%d k_base=%d src0=%p%s\n",
                g_threadCtx.cta, g_threadCtx.tid, g_threadCtx.tid / 128, (void*)d, d->swizzle,
                rows, K, ld, c0, c1, c2, c3, m_base, k_base, (void*)gbase,
                bad ? "  <<< GARBAGE" : "");
            fflush(stderr);
        }
    }

    // ---- Plain-linear path (raw cp.async.bulk.tensor, SWIZZLE_NONE) -------
    // When the CUtensorMap uses CU_TENSOR_MAP_SWIZZLE_NONE and a (rows x cols_k)
    // box, the TMA lands the whole tile contiguously row-major in smem (no cute
    // software layout). One UTMALDG copies all cols_k. swizzle==4 selects this.
    // This is the 4D-multitile coord path: each call's (k_base,m_base) selects a
    // different global tile, copied to the same smem region.
    if (d->swizzle == 4) {
        // The TMA dest is smem_base + smem_off (the [URdst] the kernel computed),
        // not tile_base_off — the whole box lands contiguously there.
        uint8_t* sdst = smem_base + smem_off;
        for (int m = 0; m < rows; ++m) {
            for (int k = 0; k < K; ++k) {
                int g_elem = (m_base + m) * ld + (k_base + k);
                const uint8_t* src = gbase + (size_t)g_elem * eb;
                std::memcpy(sdst + (size_t)(m * K + k) * eb, src, eb);
            }
        }
        // DEBUG (gated SLIFTER_LSE_TRACE): if this linear TMA tile lands in the
        // LSE/dPsum window, record it in the unified seq stream so a TMA clobber of
        // the LSE buffer is caught. Reads live smem (post-write) -> detects overlap.
        if (lse_trace_on() && g_threadCtx.cta == 0) {
            uint32_t off = (uint32_t)smem_off, nb = (uint32_t)((int64_t)rows * K * eb);
            if (off < LSE_TRACE_HI && off + nb > LSE_TRACE_LO) {
                long seq = g_lse_seq.fetch_add(1);
                float v0=0.f; std::memcpy(&v0, smem_base+smem_off, sizeof(float));
                std::fprintf(stderr, "[seq %ld] [utma] cta=0 tid=%d off=0x%x bytes=%u v0=%g\n",
                             seq, g_threadCtx.tid, off, nb, v0);
                lse_trace_record_write(off, nb, nullptr, 't', seq);
            }
        }
        slifter_mbar_deliver_tx((int64_t)rows * K * eb);  // retire expect-tx bytes
        return;
    }

    // ---- SW128 path (FA's Q/K/V layout) ----------------------------------
    // The CUtensorMap encodes the whole K_total-wide swizzled tile, so cute
    // issues ONE UTMALDG per tensor (no K-blocking). Copy the entire tile at
    // its row-major-then-swizzled physical offset.
    if (d->swizzle == 3) {
        const int maxr = d->max_rows;   // 0 => no clamp
        // Dest base = this load's smem offset ([URdst]=smem_off) = the same smem
        // address the WGMMA descriptor's start_address points at, so each tensor
        // (Q/K/V) lands at its OWN offset (the old tile_base_off=0 made them clobber).
        uint8_t* sw_base = smem_base + smem_off;
        // SW128's swizzle atom is 128 BYTES wide: 64 elements for fp16/bf16
        // (2B/elem), 128 elements for fp8 e4m3 (1B/elem). cute lays a (rows x K)
        // K-major tile as ceil(K/ATOM) atoms, each `rows x ATOM` and internally
        // swizzled by slifter_sw128_elem(m*ATOM + kin); atom `a` starts at
        // a*(rows*ATOM) elements. The WGMMA SS-decode reads with K_total=ATOM and
        // the descriptor's per-atom base, so the write MUST tile the same way.
        // (For K<=ATOM this is a single atom == slifter_sw128_off(m,k,ATOM).)
        // Probed exact vs cute: fp8 K-atom = 128 elems; fp16 K-atom = 8<<B
        // (SW128 B3->64, SW64 B2->32, SW32
        // B1->16). swiz_b==0 (every existing driver) is treated as B=3 so the
        // SW128 atom stays 64 -> byte-identical. fp8 keeps the fixed 128-wide
        // K-atom regardless of B (its 1-byte e4m3 layout is a separate path).
        const int swB  = d->swiz_b ? d->swiz_b : 3;
        const int ATOM = (eb == 1) ? 128 : (8 << swB);
        if (getenv("SLIFTER_TMA_DEBUG")) {
            static std::atomic<int> nd{0};
            if (nd.fetch_add(1) < 40) {
                int maxs;
                if (d->vt_load) {
                    maxs = 0;
                    for (int m=0;m<rows;m++) for(int k=0;k<K;k++){ int e=slifter_vt_load_elem(k,m); if(e>maxs)maxs=e; }
                } else {
                    maxs = (K/ATOM ? (K/ATOM-1):0)*(rows*ATOM) + slifter_swizzle_elem((rows-1)*ATOM + (ATOM-1), swB);
                }
                fprintf(stderr, "[utma_sw128] smem_off=%d rows=%d K=%d ATOM=%d vt_load=%d gbase=%p desc=%p -> end=%d\n",
                        smem_off, rows, K, ATOM, (int)d->vt_load, (void*)d->global_base, (void*)desc_addr,
                        smem_off + maxs*eb);
            }
        }
        // DEBUG (gated SLIFTER_KID_INJECT): inject a source-id into the hd96 K tile
        // (uniquely swB==1 && K==16) so the dq_kread B-read can be decoded NOISE-FREE:
        // each K cell holds the RAW uint16 bits ((key<<7)|hd). The B-read fpext's those
        // bits to f32; re-encoding the f32 to fp16 bits recovers the id, revealing the
        // (key,hd) the read ACTUALLY landed on regardless of value collisions. K-tile
        // only; every other tile keeps the real value so the kernel still runs. getenv
        // hoisted out of the per-cell loop (one check per tile load).
        const bool kid_inj = (swB == 1 && K == 16 && getenv("SLIFTER_KID_INJECT"));
        for (int m = 0; m < rows; ++m) {
            const bool oob = (maxr > 0 && (m_base + m) >= maxr);  // partial last K-tile
            for (int k = 0; k < K; ++k) {
                int s_elem;
                if (d->vt_load) {
                    // FA3 FP8 V load: cute's SmemLayoutVt — m is the seqlen index
                    // (the tile rows = kBlockN seqlen), k is the hdim index. The
                    // in-kernel LDSM.T reads back exactly this layout.
                    s_elem = slifter_vt_load_elem(/*hdim=*/k, /*seqlen=*/m);
                } else {
                    const int atom = k / ATOM, kin = k % ATOM;
                    s_elem = atom * (rows * ATOM) + slifter_swizzle_elem(m * ATOM + kin, swB);
                }
                const int s_byte = s_elem * eb;
                if (oob) { std::memset(sw_base + s_byte, 0, eb); continue; }  // FA masks these rows
                if (kid_inj) {
                    int gk = m_base + m, gh = k_base + k;
                    uint16_t id = (uint16_t)(((gk & 0x1ff) << 7) | (gh & 0x7f));
                    std::memcpy(sw_base + s_byte, &id, 2);
                    continue;
                }
                int g_elem = (m_base + m) * ld + (k_base + k);
                const uint8_t* src = gbase + (size_t)g_elem * eb;
                std::memcpy(sw_base + s_byte, src, eb);
            }
        }
        // DEBUG (no behavior change): dump the first rows==128 SW128 load (Q tile)
        // once so the Q-load write layout can be checked vs expected SW128 offline
        // (Q is stable after load -> no race). Identifies wrapper Q-load vs lifter
        // SS-read as the source of the seq-correlated QK residual.
        if (getenv("SLIFTER_Q_SMEM_DUMP") && rows == 128) {
            // dump the first rows==128 (Q) load for cta0 AND cta1 (m_block0/1) to
            // separate files, to verify the nonzero-seq-coord (c1=128) Q load.
            int qc = g_threadCtx.cta;
            if (qc == 0 || qc == 1) {
                static std::atomic<int> qd0{0}, qd1{0};
                bool first = (qc == 0) ? (qd0.fetch_add(1) == 0) : (qd1.fetch_add(1) == 0);
                if (first) {
                    char nm[256];
                    snprintf(nm, sizeof(nm), "q_smem_cta%d.bin", qc);
                    FILE* f = fopen(slifter_debug_path(nm).c_str(), "wb");
                    if (f) { fwrite(sw_base, eb, (size_t)rows * ATOM, f); fclose(f); }
                }
            }
        }
        // V-region dump (rows==176): with V=ones, a fully-loaded region is all 1.0;
        // partial => the V load is the coverage bug (else it's downstream wgmma/epilogue).
        if (getenv("SLIFTER_V_SMEM_DUMP") && rows == 176 && g_threadCtx.cta == 1
            && smem_off == 2048) {
            static std::atomic<int> vd{0};
            if (vd.fetch_add(1) == 0) {
                FILE* f = fopen(slifter_debug_path("v_smem.bin").c_str(), "wb");
                if (f) { fwrite(sw_base, eb, (size_t)rows * ATOM, f); fclose(f); }
                fprintf(stderr, "[v_smem] dumped rows=%d ATOM=%d (=%d elems) @smem_off=2048\n",
                        rows, ATOM, rows*ATOM);
            }
        }
        // DEBUG (gated SLIFTER_LSE_TRACE): SW128 tile -> record overlap with the
        // LSE/dPsum window in the unified seq stream (catches a TMA clobber of LSE).
        if (lse_trace_on() && g_threadCtx.cta == 0) {
            uint32_t off = (uint32_t)smem_off, nb = (uint32_t)((int64_t)rows * K * eb);
            if (off < LSE_TRACE_HI && off + nb > LSE_TRACE_LO) {
                long seq = g_lse_seq.fetch_add(1);
                float v0=0.f; std::memcpy(&v0, sw_base, sizeof(float));
                std::fprintf(stderr, "[seq %ld] [utma_sw] cta=0 tid=%d off=0x%x bytes=%u v0=%g\n",
                             seq, g_threadCtx.tid, off, nb, v0);
                lse_trace_record_write(off, nb, nullptr, 't', seq);
            }
        }
        slifter_mbar_deliver_tx((int64_t)rows * K * eb);  // retire expect-tx bytes
        return;
    }

    // ---- K-INTER (no-swizzle) path ---------------------------------------
    // K-INTER issues one UTMALDG per 8-wide K-block. Recover this call's block
    // index from its smem offset relative to the tile base, and copy only that
    // block's 8 global K-columns into the block's own smem region.
    const int kblock_bytes = rows * 8 * eb;          // one K-block's smem span
    int block = 0;
    if (kblock_bytes > 0)
        block = (smem_off - d->tile_base_off) / kblock_bytes;
    if (block < 0) block = 0;
    const int k_lo = block * 8;
    int k_hi = k_lo + 8;
    if (k_hi > K) k_hi = K;

    uint8_t* sdst = smem_base + smem_off;   // == tile_base + block*kblock_bytes
    for (int m = 0; m < rows; ++m) {
        for (int k = k_lo; k < k_hi; ++k) {
            // Global element (row-major, leading dim ld).
            int g_elem = (m_base + m) * ld + (k_base + k);
            const uint8_t* src = gbase + (size_t)g_elem * eb;
            // Shared element within this block: K-INTER local offset (k in
            // [0,8) inside the block).
            int s_elem = slifter_kinter_off(m, k - k_lo, rows);
            std::memcpy(sdst + (size_t)s_elem * eb, src, eb);
        }
    }
    slifter_mbar_deliver_tx((int64_t)rows * (k_hi - k_lo) * eb);  // retire expect-tx bytes
}

// TMA bulk STORE: shared -> global, the inverse of utma_load. Used by the
// FlashAttention epilogue (UTMASTG.5D) to write the computed O tile out to the
// global O tensor. Operand roles mirror the load: the descriptor is the O
// CUtensorMap-shaped SlifterTmaDesc, smem_off is the byte offset of the O tile
// in shared, and {c0..c3} are the recovered output-tile coordinates
//   c0 = headdim-mode base (0 for the single hdim tile)
//   c1 = seq-tile ROW base (which query rows this CTA wrote)
//   c2,c3 = head/batch, folded into the global base.
// The smem O tile holds fp16 in the descriptor's swizzle layout (same physical
// formula as the load); this copies each logical (m,k) out to global row-major.
extern "C" void utma_store(uint64_t desc_addr, uint8_t* smem_base,
                           int32_t smem_off,
                           int32_t c0, int32_t c1, int32_t c2, int32_t c3)
{
    if (getenv("SLIFTER_TMA_DEBUG") && g_threadCtx.lane == 0) {
        static std::atomic<int> ns{0};
        if (ns.fetch_add(1) < 400) {
            const SlifterTmaDesc* dd = reinterpret_cast<const SlifterTmaDesc*>(desc_addr);
            std::fprintf(stderr, "[utma_store] cta=%d tid=%d wg=%d smem_off=%d c0=%d c1=%d c2=%d c3=%d gbase=%p store_dkv=%d\n",
                         g_threadCtx.cta, g_threadCtx.tid, g_threadCtx.tid/128,
                         smem_off, c0, c1, c2, c3, dd?dd->global_base:nullptr, dd?dd->store_dkv:-1);
        }
    }
    // Warp-uniform single-issue (see utma_load): the TMA store is issued from
    // the elected (lowest) lane whose uniform registers hold valid coords.
    if (g_threadCtx.lane != 0) return;
    const SlifterTmaDesc* d =
        reinterpret_cast<const SlifterTmaDesc*>(desc_addr);
    if (!d || !d->global_base) return;
    const int rows = d->rows, K = d->cols_k, eb = d->elem_bytes;
    const int ld = d->ld ? d->ld : K;
    int k_base = c0;
    int m_base = c1;
    // Per-mode outer strides (see utma_load): c2=head*stride_c2, c3=batch*stride_c3.
    long long outer_elems =
        (long long)c2 * (long long)d->stride_c2 +
        (long long)c3 * (long long)d->stride_c3;
    uint8_t* gbase = const_cast<uint8_t*>(
        reinterpret_cast<const uint8_t*>(d->global_base)) + (size_t)outer_elems * eb;
    uint8_t* sbase = smem_base + d->tile_base_off;

    // Plain-linear (SWIZZLE_NONE) — whole box contiguous row-major at smem_off.
    if (d->swizzle == 4) {
        uint8_t* ssrc = smem_base + smem_off;
        for (int m = 0; m < rows; ++m)
            for (int k = 0; k < K; ++k) {
                int g_elem = (m_base + m) * ld + (k_base + k);
                std::memcpy(gbase + (size_t)g_elem * eb,
                            ssrc + (size_t)(m * K + k) * eb, eb);
            }
        return;
    }
    // SW128 (FA's O layout). The O-store TMA coordinate vector is (seq, head,
    // batch) — DIFFERENT mode order from the Q/K/V load (headdim, seq, head,
    // batch): the store DROPS the headdim mode (it's the inner per-element dim,
    // iterated by k), so c0=seq-tile base, c1=head, c2=batch (c3 unused, zeroed
    // by the kernel). Verified by the per-CTA store-coord dump (grid=16):
    //   c0 in {0,128}=m_block*128 (seq), c1 in {0..3}=head, c2 in {0,1}=batch.
    // Each epilogue issues one UTMASTG per headdim ATOM (64 wide); the two atoms
    // land at smem_off 0 and rows*64*eb apart, so atom = smem_off/(rows*64*eb)
    // selects the global headdim half (d_base = atom*64). The (head,batch) plane
    // is folded into the global base with the same per-mode strides as the load
    // (stride_c2=head=D, stride_c3=batch=SQ*H*D).
    if (d->swizzle == 3) {
        // Per-descriptor swizzle bit-width (default 3 == SW128, byte-identical to
        // every existing store driver which leaves swiz_b=0). ATOM = 8<<B; the
        // within-atom element offset uses slifter_swizzle_elem(...,swB). The dKV
        // and fwd O stores are SW128 (B3) on all verified kernels, so this stays
        // ATOM=64 unless a driver opts into a narrower atom.
        const int swB = d->swiz_b ? d->swiz_b : 3;
        const int ATOM = 8 << swB;
        const int seq_base = c0, head = c1, batch = c2;
        const int atom_span = rows * ATOM * eb;            // bytes per headdim atom in smem
        // FA3 BACKWARD dKV store: dK and dV interleave in ONE smem segment (dK @
        // region base 0, dV @ region base 0x8000), so the headdim ATOM index must
        // be RELATIVE to each tensor's smem region base — not absolute as for the
        // fwd O store (where the two O atoms live at smem 0 / 0x4000). store_dkv
        // encodes (region_base_bytes + 1); 0 => fwd O-store (unchanged below).
        // The WITHIN-ATOM (key,hd)->smem element map is IDENTICAL to the fwd O
        // store: the dKV R2S copy is the same STSM_N atom (dKV_swapAB=false), which
        // writes the same physical mr/corr bit-swizzle on top of plain SW128.
        // Value-verified: with the fwd mr/corr map dV atom-0 corr=1.0000 vs CPU
        // dV; plain SW128 gives 0.63.
        if (d->store_dkv) {
            const int region_base = d->store_dkv - 1;
            const int atom = atom_span ? ((smem_off - region_base) / atom_span) : 0;
            const int d_base = atom * ATOM;
            uint8_t* obase = const_cast<uint8_t*>(
                reinterpret_cast<const uint8_t*>(d->global_base))
                + (size_t)((long long)head * d->stride_c2
                           + (long long)batch * d->stride_c3) * eb;
            uint8_t* ssrc = smem_base + smem_off;          // this atom's swizzled smem
            // DEBUG (gated, no behavior change): raw dV/dK atom-0 smem dump for the
            // first (cta0, seq-tile0, head0, batch0) store so the true smem->(key,hd)
            // map can be value-checked offline against CPU dV/dK.
            if (getenv("SLIFTER_DV_SMEM_DUMP") && g_threadCtx.cta == 0 && seq_base == 0
                && head == 0 && batch == 0 && atom == 0) {
                static std::atomic<int> dvd{0};
                int which = (region_base != 0) ? 1 : 0;   // 0=dK, 1=dV
                if (dvd.fetch_add(1) < 2) {
                    char path[256];
                    std::snprintf(path, sizeof(path), "%s_smem_full.bin",
                                  which ? "dv" : "dk");
                    FILE* f = fopen(slifter_debug_path(path).c_str(), "wb");
                    // Full dKV smem region (debug): dV region base 0x6000 (12288 elems for
                    // hd96 SW64 = 3 atoms * 128 keys * 32). Dump 12288 elems from region base.
                    uint8_t* rbase = smem_base + region_base;
                    if (f) { fwrite(rbase, 1, (size_t)12288 * eb, f); fclose(f); }
                }
            }
            for (int m = 0; m < rows; ++m) {
                const int mr   = m ^ ((m >> 3) & 1);
                const int corr = 8 * ((mr & 1) ^ ((mr >> 3) & 1));
                for (int k = 0; k < ATOM; ++k) {
                    long long g_elem = (long long)(seq_base + m) * ld + (d_base + k);
                    // SW64 (B=2) dKV tile: STSM_N partition_D map (cute Sw<2,4,3>), m=key,
                    // k=hd within the 32-wide atom (ssrc already at the atom's smem base).
                    // SW128 (B=3: hd64/128/192/256) keeps the byte-identical mr/corr path.
                    int s_elem = (swB == 2)
                        ? slifter_dkv_stsm_sw64(m, k)
                        : slifter_swizzle_elem(mr * ATOM + (k ^ corr), swB);
                    std::memcpy(obase + (size_t)g_elem * eb,
                                ssrc + (size_t)s_elem * eb, eb);
                }
            }
            return;
        }
        const int atom = atom_span ? (smem_off / atom_span) : 0;
        const int d_base = atom * ATOM;
        uint8_t* obase = const_cast<uint8_t*>(
            reinterpret_cast<const uint8_t*>(d->global_base))
            + (size_t)((long long)head * d->stride_c2
                       + (long long)batch * d->stride_c3) * eb;
        uint8_t* ssrc = smem_base + smem_off;              // this atom's swizzled smem
        // DEBUG (no behavior change): dump the raw atom-0 smem region for the very
        // first (b0,h0,seq0) store so the true smem->logical layout can be derived
        // offline by matching values against the reference O tile.
        if (getenv("SLIFTER_O_SMEM_DUMP") && g_threadCtx.cta == 0 && seq_base == 0
            && head == 0 && batch == 0 && atom == 0) {
            static std::atomic<int> dumped{0};
            if (dumped.fetch_add(1) == 0) {
                FILE* f = fopen(slifter_debug_path("o_smem_atom0.bin").c_str(), "wb");
                if (f) { fwrite(ssrc, 1, (size_t)rows * ATOM * eb, f); fclose(f); }
            }
        }
        // DEBUG (gated, no behavior change): per-hdim-block census of what this
        // atom's smem READ region holds (the bf16 O bytes the STSM stored). For
        // V=ones every covered hdim should be ~v_descale; a block that reads all
        // zeros means the STSM never wrote that smem sub-region (write bug) — or
        // this read maps it to the wrong s_elem (read bug). atom 0 -> global hd
        // 0-63, atom 1 -> 64-127, so we tag blocks by (atom*64 + k)/32.
        if (getenv("SLIFTER_O_SMEM_CENSUS") && g_threadCtx.cta == 0 && seq_base == 0
            && head == 0 && batch == 0) {
            static std::atomic<int> cen{0};
            // one census per atom (atom 0 then atom 1)
            if (cen.fetch_add(1) < 2) {
                long long nz[4] = {0,0,0,0}, tot[4] = {0,0,0,0};
                for (int m = 0; m < rows; ++m) {
                    const int mr   = m ^ ((m >> 3) & 1);
                    const int corr = 8 * ((mr & 1) ^ ((mr >> 3) & 1));
                    for (int k = 0; k < ATOM; ++k) {
                        int s_elem = slifter_swizzle_elem(mr * ATOM + (k ^ corr), swB);
                        uint16_t bits;
                        std::memcpy(&bits, ssrc + (size_t)s_elem * eb, 2);
                        int blk = (atom * ATOM + k) / 32;   // 0..3 over global hdim
                        tot[blk]++; if (bits) nz[blk]++;
                    }
                }
                std::fprintf(stderr, "[o_smem_census] atom=%d smem_off=%d  blk0 %lld/%lld | blk1 %lld/%lld | blk2 %lld/%lld | blk3 %lld/%lld\n",
                             atom, smem_off, nz[0],tot[0], nz[1],tot[1], nz[2],tot[2], nz[3],tot[3]);
                // also census the RAW smem region this atom occupies (no swizzle map)
                long long rnz = 0, rtot = (long long)rows * ATOM;
                for (long long i = 0; i < rtot; ++i) {
                    uint16_t b; std::memcpy(&b, ssrc + i*eb, 2); if (b) rnz++;
                }
                std::fprintf(stderr, "[o_smem_census] atom=%d RAW nonzero=%lld/%lld\n", atom, rnz, rtot);
            }
        }
        for (int m = 0; m < rows; ++m) {
            // STSM.M88 stores the O tile with two bit swizzles beyond plain SW128,
            // both decoded from probes (V=hd for hd, identity-attention P=I for seq):
            //  * SEQ: logical seq m lives at physical smem row m ^ ((m>>3)&1)
            //    (decoded: P=I gives O[seq]=seq; rows 8<->9,10<->11,24<->25,... swap
            //    when seq bit3 is set).
            //  * HD: within physical row, headdim k lives at k ^ corr, corr =
            //    8*((mr&1) ^ ((mr>>3)&1)) keyed on the PHYSICAL row mr (V=hd probe).
            const int mr   = m ^ ((m >> 3) & 1);                 // physical smem row
            const int corr = 8 * ((mr & 1) ^ ((mr >> 3) & 1));
            for (int k = 0; k < ATOM; ++k) {
                long long g_elem = (long long)(seq_base + m) * ld + (d_base + k);
                int s_elem = slifter_swizzle_elem(mr * ATOM + (k ^ corr), swB);
                std::memcpy(obase + (size_t)g_elem * eb,
                            ssrc + (size_t)s_elem * eb, eb);
            }
        }
        return;
    }
    // K-INTER (no-swizzle) per-K-block, recovered from smem_off.
    const int kblock_bytes = rows * 8 * eb;
    int block = (kblock_bytes > 0) ? (smem_off - d->tile_base_off) / kblock_bytes : 0;
    if (block < 0) block = 0;
    const int k_lo = block * 8;
    int k_hi = k_lo + 8; if (k_hi > K) k_hi = K;
    uint8_t* ssrc = smem_base + smem_off;
    for (int m = 0; m < rows; ++m)
        for (int k = k_lo; k < k_hi; ++k) {
            int g_elem = (m_base + m) * ld + (k_base + k);
            int s_elem = slifter_kinter_off(m, k - k_lo, rows);
            std::memcpy(gbase + (size_t)g_elem * eb, ssrc + (size_t)s_elem * eb, eb);
        }
}

// UBLKRED.G.S.ADD.F32 — TMA bulk REDUCE-add: gmem[addr..addr+size] +=
// smem[smem_base+smem_off..+size], element-wise fp32. FA3 backward dQ_accum
// accumulation. This is a FLAT 1D reduce (NOT tensor-tiled): the smem->gmem
// correspondence is identity over `size_bytes`. Warp-uniform single-issue (the
// UBLKRED operands are uniform UR registers), so it fires once from lane 0.
// Atomic fadd handles the per-thread x86 backend running CTAs concurrently AND
// multiple bulk-reduces targeting overlapping dQ_accum regions (deterministic
// mode); under fibers each is serialized so the atomic is a correctness no-op.
extern "C" void ublk_reduce_add(uint64_t gmem_addr, uint8_t* smem_base,
                                int32_t smem_off, int64_t size_bytes)
{
    if (getenv("SLIFTER_UBLKRED_DEBUG") && g_threadCtx.lane == 0) {
        static std::atomic<int> ns{0};
        if (ns.fetch_add(1) < 200)
            std::fprintf(stderr, "[ublkred] cta=%d tid=%d wg=%d gmem=%p smem_off=%d size=%lld floats=%lld\n",
                         g_threadCtx.cta, g_threadCtx.tid, g_threadCtx.tid/128,
                         (void*)gmem_addr, smem_off, (long long)size_bytes,
                         (long long)(size_bytes/4));
    }
    if (g_threadCtx.lane != 0) return;
    if (!gmem_addr || size_bytes <= 0) return;
    float* gdst = reinterpret_cast<float*>(gmem_addr);
    const float* ssrc = reinterpret_cast<const float*>(smem_base + (smem_off & 0xFFFFFF));
    const int64_t n = size_bytes / 4;
    // DIAG (gated SLIFTER_UBLKRED_TILE): dump each cta0 reduce's smem source tile +
    // its gmem byte offset so the dQ GEMM->smem result can be value-checked vs CPU
    // dQ_accum BEFORE the cross-block atomic accumulation. Runtime-gated -> no prod
    // effect; the .o is unchanged (wrapper-only rebuild).
    if (getenv("SLIFTER_UBLKRED_TILE") && g_threadCtx.cta == 0) {
        static FILE* tf = nullptr; static std::mutex tmx; std::lock_guard<std::mutex> lk(tmx);
        if (!tf) tf = fopen(slifter_debug_path("ublkred_tile.bin").c_str(), "wb");
        if (tf) {
            // record: int64 gmem_addr, int32 wg, int32 nfloats, then nfloats f32
            int32_t wg = g_threadCtx.tid/128, nf = (int32_t)n;
            fwrite(&gmem_addr,8,1,tf); fwrite(&wg,4,1,tf); fwrite(&nf,4,1,tf);
            fwrite(ssrc,4,n,tf); fflush(tf);
        }
    }
    for (int64_t i = 0; i < n; ++i) {
        std::atomic_ref<float> a(gdst[i]);
        float cur = a.load(std::memory_order_relaxed);
        while (!a.compare_exchange_weak(cur, cur + ssrc[i],
                                        std::memory_order_relaxed)) {}
    }
}

extern "C" int32_t shfl_down_i32(uint32_t mask,
                                 int32_t  val,
                                 int32_t  offset,
                                 int32_t  clamp,
                                 uint32_t site)
{
    return warp_shfl_down<int32_t>(mask, val, offset, clamp, site);
}

extern "C" float shfl_down_f32(uint32_t mask,
                               float    val,
                               int32_t  offset,
                               int32_t  clamp,
                               uint32_t site)
{
    return warp_shfl_down<float>(mask, val, offset, clamp, site);
}

extern "C" int32_t shfl_up_i32(uint32_t mask,
                               int32_t  val,
                               int32_t  offset,
                               int32_t  clamp,
                               uint32_t site)
{
    return warp_shfl_up<int32_t>(mask, val, offset, clamp, site);
}

extern "C" float shfl_up_f32(uint32_t mask,
                             float    val,
                             int32_t  offset,
                             int32_t  clamp,
                             uint32_t site)
{
    return warp_shfl_up<float>(mask, val, offset, clamp, site);
}

// shfl_idx: direct lane index shuffle (warp-synchronized)
template<typename T>
T warp_shfl_idx(uint32_t mask, T val, int32_t srcLane, int32_t clamp,
                uint32_t site)
{
    (void)mask;
    static_assert(sizeof(T) == 4, "warp_shfl_idx only supports 32-bit types");
    const int lane         = g_threadCtx.lane;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return val;
    const int width = shfl_effective_width(clamp, lanesInWarp);

    uint32_t raw; std::memcpy(&raw, &val, 4);
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    // A full-warp mask (0xffffffff) is a reconvergence point: gather all live warp
    // lanes structurally, not by divergence-history (which fragments warp-spec'd FA
    // consumer lanes into singletons -> SHFL.IDX returns own value). Same fix as
    // warp_shfl_bfly (commit ea41b919); without it the packgqa O-store SHFL-scatter
    // collapsed (4 srcLane-iters all returned own value -> overwrite -> 3/4 of O zero).
    const uint32_t members = warp_group_gather(site, raw, true, vals, preds,
                                               mask == 0xffffffffu, lanesInWarp);

    if (getenv("SLIFTER_SHFLIN_DBG") && clamp == 0x181f && g_threadCtx.cta == 0
        && g_threadCtx.warpInBlock == 4) {
        static std::atomic<int> ns{0};
        if (ns.fetch_add(1) < 64)
            std::fprintf(stderr, "[shflin] warp=%d lane=%d srcLane=%d val=%u (members=0x%08x)\n",
                         g_threadCtx.warpInBlock, lane, srcLane, raw, members);
    }
    const int src = srcLane;
    uint32_t outRaw = raw;
    // HW SHFL.IDX honors a segmentation mask in c[12:8] (segmask): each lane
    // reads from src = (laneid & segmask) | (srcLane & ~segmask), i.e. the
    // shuffle is segment-relative (segments of ~segmask+1 lanes). The width
    // path below (segmask==0) treats c[4:0] as a plain width; it must NOT be
    // used when a genuine partial segmask is present, or every segment would
    // collapse onto global lanes 0..(width-1). The packgqa O-store scatter uses
    // clamp 0x181f -> segmask 0x18 (8-lane segments); without segment-relative
    // addressing all 4 segments hit lanes 0..3 -> only 1/4 of O rows written.
    // segmask==0x1f (e.g. CUB c==-1 partial-tile sentinel) is degenerate (each
    // lane its own segment); keep it on the width path = full warp.
    const int32_t segmask = (clamp >> 8) & 0x1f;
    if (segmask != 0 && segmask != 0x1f) {
        const int minLane  = lane & segmask;
        const int clampVal = clamp & 0x1f;
        const int maxLane  = minLane | (clampVal & ~segmask);
        const int j        = minLane | (src & ~segmask);
        if (j <= maxLane && j < WARP_SIZE && ((members >> j) & 1))
            outRaw = vals[j];
    } else {
        const int seg = (lane / width) * width;
        if (src >= seg && src < seg + width && src < WARP_SIZE
            && ((members >> src) & 1))
            outRaw = vals[src];
    }
    T out; std::memcpy(&out, &outRaw, sizeof(T));
    return out;
}

extern "C" int32_t shfl_idx_i32(uint32_t mask, int32_t val, int32_t srcLane, int32_t clamp, uint32_t site)
{
    return warp_shfl_idx<int32_t>(mask, val, srcLane, clamp, site);
}

extern "C" float shfl_idx_f32(uint32_t mask, float val, int32_t srcLane, int32_t clamp, uint32_t site)
{
    return warp_shfl_idx<float>(mask, val, srcLane, clamp, site);
}

// shfl_bfly: butterfly (XOR) shuffle (warp-synchronized)
template<typename T>
T warp_shfl_bfly(uint32_t mask, T val, int32_t laneMask, int32_t clamp,
                 uint32_t site)
{
    static_assert(sizeof(T) == 4, "warp_shfl_bfly only supports 32-bit types");
    const int lane         = g_threadCtx.lane;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return val;
    const int width = shfl_effective_width(clamp, lanesInWarp);

    uint32_t raw; std::memcpy(&raw, &val, 4);
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    // A full-warp mask (0xffffffff) is a reconvergence point: gather all live
    // warp lanes structurally, not by divergence-history (which fragments
    // warp-spec'd consumer lanes into singletons -> broken reduction).
    const uint32_t members = warp_group_gather(site, raw, true, vals, preds,
                                               mask == 0xffffffffu, lanesInWarp);

    int src = lane ^ laneMask;
    const int seg = (lane / width) * width;
    uint32_t outRaw = raw;
    bool partner_in_range = (src >= seg && src < seg + width && src < WARP_SIZE);
    bool partner_member = ((members >> src) & 1);
    if (partner_in_range && partner_member)
        outRaw = vals[src];
    if (getenv("SLIFTER_BFLY_DEBUG") && lane < 4 && g_threadCtx.warpInBlock==4) {
        static std::atomic<int> nb{0};
        if (nb.fetch_add(1) < 40)
            fprintf(stderr, "[bfly] cta=%d warp=%d lane=%d laneMask=%d members=0x%08x popcount=%d "
                    "partner_member=%d\n",
                    g_threadCtx.cta, g_threadCtx.warpInBlock, lane, laneMask, members,
                    __builtin_popcount(members), partner_member);
    }
    T out; std::memcpy(&out, &outRaw, sizeof(T));
    return out;
}

extern "C" int32_t shfl_bfly_i32(uint32_t mask, int32_t val, int32_t laneMask, int32_t clamp, uint32_t site)
{
    return warp_shfl_bfly<int32_t>(mask, val, laneMask, clamp, site);
}

extern "C" float shfl_bfly_f32(uint32_t mask, float val, int32_t laneMask, int32_t clamp, uint32_t site)
{
    return warp_shfl_bfly<float>(mask, val, laneMask, clamp, site);
}

// -----------------------------------------------------------------------------
// REDUX (sm80+): warp-wide reduction into a uniform register. Every live
// lane of the history-matched group receives the reduction of all members'
// payloads. NOT an identity — nvcc's atomic aggregation emits
//   REDUX.SUM UR, Rv ; VOTEU.ANY/UFLO leader election ; @leader REDG UR
// so an identity REDUX silently drops every non-leader lane's contribution
// (hecbench mr-cuda: val stayed 0 because the leader lane's own verdict was 0).
// -----------------------------------------------------------------------------
template<typename Op>
static uint32_t warp_redux(uint32_t site, uint32_t raw, Op&& op)
{
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return raw;

    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, raw, true, vals, preds);

    uint32_t acc = 0; bool first = true;
    for (int M = 0; M < WARP_SIZE; ++M) {
        if (!((members >> M) & 1)) continue;
        acc = first ? vals[M] : op(acc, vals[M]);
        first = false;
    }
    return acc;
}

extern "C" uint32_t redux_sum(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a + b; });
}
extern "C" uint32_t redux_and(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a & b; });
}
extern "C" uint32_t redux_or(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a | b; });
}
extern "C" uint32_t redux_xor(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a ^ b; });
}
extern "C" uint32_t redux_min_u32(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a < b ? a : b; });
}
extern "C" uint32_t redux_max_u32(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) { return a > b ? a : b; });
}
extern "C" uint32_t redux_min_s32(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) {
        return (int32_t)a < (int32_t)b ? a : b; });
}
extern "C" uint32_t redux_max_s32(uint32_t v, uint32_t site)
{
    return warp_redux(site, v, [](uint32_t a, uint32_t b) {
        return (int32_t)a > (int32_t)b ? a : b; });
}

// half-precision helpers
//
// The lifter declares hmax2/hmin2 as `<2 x half> (<2 x half>, <2 x half>)`
// (HMNMX2). g++ 11 has no _Float16, so these use `float` as the ABI
// container: both are 4-byte SSE-class values (xmm0/xmm1 args, xmm0
// return) under SysV x86-64, and only the bit pattern matters.
static inline float h2f_bits(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t man  = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {  // subnormal: renormalize
            int shift = 0;
            while (!(man & 0x400)) { man <<= 1; shift++; }
            out = sign | ((uint32_t)(113 - shift) << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (man << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f; std::memcpy(&f, &out, 4); return f;
}

static inline uint16_t f2h_bits(float f)
{
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    uint32_t e8   = (x >> 23) & 0xff;
    uint32_t man  = x & 0x7fffff;
    if (e8 == 0xff)  // inf / nan
        return (uint16_t)(sign | 0x7c00 | (man ? 0x200 | (man >> 13) : 0));
    int32_t exp = (int32_t)e8 - 127 + 15;
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00);  // overflow -> inf
    if (exp <= 0) {                                     // half subnormal/zero
        if (exp < -11) return (uint16_t)sign;
        man |= 0x800000;
        int shift = 14 - exp;  // value = man * 2^(exp-14) in subnormal units
        uint32_t sub  = man >> shift;
        uint32_t rem  = man & ((1u << shift) - 1);
        uint32_t halfu = 1u << (shift - 1);
        if (rem > halfu || (rem == halfu && (sub & 1))) sub++;
        return (uint16_t)(sign | sub);
    }
    uint16_t h = (uint16_t)(sign | ((uint32_t)exp << 10) | (man >> 13));
    uint32_t rem = man & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;  // RNE; carry is correct
    return h;
}

template <float Sel(float, float)>
static inline float h2_lanewise(float a, float b)
{
    uint32_t ua, ub;
    std::memcpy(&ua, &a, 4); std::memcpy(&ub, &b, 4);
    uint16_t r0 = f2h_bits(Sel(h2f_bits((uint16_t)ua), h2f_bits((uint16_t)ub)));
    uint16_t r1 = f2h_bits(Sel(h2f_bits((uint16_t)(ua >> 16)),
                               h2f_bits((uint16_t)(ub >> 16))));
    uint32_t ur = (uint32_t)r0 | ((uint32_t)r1 << 16);
    float r; std::memcpy(&r, &ur, 4); return r;
}

// PTX max/min.f16x2 without .NaN follow IEEE maxNum/minNum (return the
// non-NaN operand) — same as fmaxf/fminf.
extern "C" float hmax2(float a, float b)     { return h2_lanewise<fmaxf>(a, b); }
extern "C" float hmax2_ftz(float a, float b) { return h2_lanewise<fmaxf>(a, b); }
extern "C" float hmin2(float a, float b)     { return h2_lanewise<fminf>(a, b); }
extern "C" float hmin2_ftz(float a, float b) { return h2_lanewise<fminf>(a, b); }

// bf16 truncation libcall: clang lowers `fptrunc float to bfloat` to
// __truncsfbf2 on generic x86, but g++ 11's libgcc predates it. __bf16
// is SSE-class like float, so a float container with the bf16 pattern
// in its low 16 bits satisfies the ABI.
extern "C" float __truncsfbf2(float x)
{
    uint32_t bits; std::memcpy(&bits, &x, 4);
    uint16_t bf;
    if ((bits & 0x7fffffffu) > 0x7f800000u)
        bf = (uint16_t)((bits >> 16) | 0x0040);  // quiet the NaN
    else
        bf = (uint16_t)((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);  // RNE
    uint32_t out = bf;
    float r; std::memcpy(&r, &out, 4); return r;
}

// -----------------------------------------------------------------------------
// MUFU math functions
// -----------------------------------------------------------------------------

extern "C" float rcp_f32(float x) {
    if (getenv("SLIFTER_RCP_DEBUG")) {
        // Dump rcp inputs for the consumer warpgroups (tid>=128, lane 0) — the
        // softmax 1/l reciprocal. If warp-0's l is huge, 1/l~0 -> O[q0-15]=0.
        static std::atomic<int> nr{0};
        if (g_threadCtx.tid >= 128 && g_threadCtx.lane == 0 && nr.fetch_add(1) < 40)
            std::fprintf(stderr, "[rcp] tid=%d wg=%d warp=%d x(=l?)=%.4f 1/x=%.5f\n",
                         g_threadCtx.tid, g_threadCtx.tid/128, (g_threadCtx.tid%128)/32, x, 1.0f/x);
    }
    // MUFU.RCP on NVIDIA GPUs returns a reciprocal approximation (~23 bits of
    // mantissa precision).  It is NOT IEEE-754 correctly-rounded division.
    // The hardware result has a slight upward magnitude bias compared to the
    // exact mathematical reciprocal — even for exact cases like powers of 2.
    //
    // NVIDIA's software integer-division sequences (compiled by nvcc) rely on
    // this upward bias: the Barrett/Newton-Raphson magic number computed from
    // MUFU.RCP must be large enough for the 2-step correction loop to converge.
    // Using exact IEEE 1.0f/x produces a magic number that is too small,
    // causing the correction loop to under-count the quotient by 1.
    //
    // Fix: bump the reciprocal magnitude upward by a few ULPs.  The GPU's
    // MUFU.RCP has documented relative error <= 2^-22.4, and empirically the
    // hardware result for powers of 2 is ~3 ULPs above the exact value.
    // We bias by 3 ULPs to match this behavior.
    float r = 1.0f / x;
    // Power-of-two inputs are EXACT on hardware MUFU.RCP — and must stay
    // exact here. Proof by counterexample for the +3 bias: cudnn's
    // uniform-register Barrett warpid chain (bn_bw reductions,
    // libcudnn_ops.so.69) computes magic = F2I(rcp(32.0f) + 0x0ffffffe)
    // and then one Newton step + a 2-step additive quotient correction.
    // That correction absorbs at most +2 ULPs of rcp error; at +3 the
    // F2I input crosses the 2^27 mantissa boundary (float ULP jumps
    // 2 -> 16), the effective magic DOUBLES (0x1000000f vs 0x07ffffff),
    // warpid becomes tid>>4, lanes go negative, only tid 0 stores its
    // warp partial, and the cross-warp reduction silently drops 15/16
    // partials (value-audit forensics: temp/bn_oracle/NOTES.md). The
    // same chain on H100 is correct, so hardware rcp(2^k) cannot carry
    // the +3 bias. Exact rcp for 2^k also fixes genuine float math
    // (1/2.0 must be 0.5, not 0.5 + 3 ULPs).
    uint32_t xbits;
    memcpy(&xbits, &x, 4);
    if ((xbits & 0x007FFFFFu) == 0)   // zero mantissa: +-2^k, +-0, +-inf
        return r;
    uint32_t bits;
    memcpy(&bits, &r, 4);
    // Bump magnitude up by 3 ULPs (handles both positive and negative)
    if (x > 0.0f)
        bits += 3;
    else if (x < 0.0f)
        bits -= 3;  // for negative floats, subtracting increases magnitude
    memcpy(&r, &bits, 4);
    return r;
}
extern "C" double rcp_f64(double x) { return 1.0 / x; }
extern "C" double rsqrt_f64(double x) { return 1.0 / sqrt(x); }
extern "C" float rsqrt_f32(float x) { return 1.0f / sqrtf(x); }
extern "C" float sin_f32(float x) { return sinf(x); }
extern "C" float cos_f32(float x) { return cosf(x); }
extern "C" float exp2_f32(float x) { return exp2f(x); }
extern "C" float log2_f32(float x) { return log2f(x); }

// -----------------------------------------------------------------------------
// Bit operations
// -----------------------------------------------------------------------------

extern "C" uint32_t brev32(uint32_t x)
{
    x = ((x >> 1) & 0x55555555) | ((x & 0x55555555) << 1);
    x = ((x >> 2) & 0x33333333) | ((x & 0x33333333) << 2);
    x = ((x >> 4) & 0x0F0F0F0F) | ((x & 0x0F0F0F0F) << 4);
    x = ((x >> 8) & 0x00FF00FF) | ((x & 0x00FF00FF) << 8);
    return (x >> 16) | (x << 16);
}

extern "C" uint64_t brev64(uint64_t x)
{
    x = ((x >> 1) & 0x5555555555555555ULL) | ((x & 0x5555555555555555ULL) << 1);
    x = ((x >> 2) & 0x3333333333333333ULL) | ((x & 0x3333333333333333ULL) << 2);
    x = ((x >> 4) & 0x0F0F0F0F0F0F0F0FULL) | ((x & 0x0F0F0F0F0F0F0F0FULL) << 4);
    x = ((x >> 8) & 0x00FF00FF00FF00FFULL) | ((x & 0x00FF00FF00FF00FFULL) << 8);
    x = ((x >> 16) & 0x0000FFFF0000FFFFULL) | ((x & 0x0000FFFF0000FFFFULL) << 16);
    return (x >> 32) | (x << 32);
}

extern "C" uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel)
{
    // Byte permutation: select bytes from a/b based on sel
    uint8_t bytes[8];
    bytes[0] = a & 0xFF; bytes[1] = (a >> 8) & 0xFF;
    bytes[2] = (a >> 16) & 0xFF; bytes[3] = (a >> 24) & 0xFF;
    bytes[4] = b & 0xFF; bytes[5] = (b >> 8) & 0xFF;
    bytes[6] = (b >> 16) & 0xFF; bytes[7] = (b >> 24) & 0xFF;
    
    uint32_t result = 0;
    for (int i = 0; i < 4; ++i) {
        uint32_t idx = (sel >> (i * 4)) & 0x7;
        result |= (uint32_t)bytes[idx] << (i * 8);
    }
    return result;
}

// -----------------------------------------------------------------------------
// match_any - warp match operation
// Matches IR:
//   declare i32 @"match_any_i32"(i32 %mask, i32 %value)
//   declare i32 @"match_any_i64"(i32 %mask, i64 %value)
// -----------------------------------------------------------------------------

// Real match: gather member lanes' values (history-matched membership, same
// machinery as vote_*), return lanes whose value equals the caller's. Result
// is ANDed with the explicit member mask — unlike votes, MATCH semantics are
// defined relative to the mask argument (sub-warp tiles pass partial masks;
// cub radix ranking peer-counts popc(result), so over-wide masks corrupt
// ranks). The old placeholder returned 0xFFFFFFFF ("everyone matches"),
// which is what made cub Onesweep ranking compute wild shared indices.
extern "C" uint32_t match_any_i32(uint32_t mask, int32_t value, uint32_t site)
{
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return (1u << g_threadCtx.lane) & mask;
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, (uint32_t)value, false, vals, preds);
    uint32_t res = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && vals[M] == (uint32_t)value) res |= (1u << M);
    return res & mask;
}

extern "C" uint32_t match_any_i64(uint32_t mask, int64_t value, uint32_t site)
{
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return (1u << g_threadCtx.lane) & mask;
    // Two 32-bit gathers (lo, hi): the shared gather payload is u32. Both are
    // deterministic same-order events on every member lane (no kernel code
    // runs between them), so membership is identical across the pair.
    const uint32_t lo = (uint32_t)((uint64_t)value & 0xFFFFFFFFu);
    const uint32_t hi = (uint32_t)((uint64_t)value >> 32);
    uint32_t vlo[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members_lo = warp_group_gather(site, lo, false, vlo, preds);
    uint32_t vhi[WARP_SIZE];
    const uint32_t members_hi = warp_group_gather(site ^ 0x9E3779B9u, hi, false, vhi, preds);
    const uint32_t members = members_lo & members_hi;
    uint32_t res = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && vlo[M] == lo && vhi[M] == hi) res |= (1u << M);
    return res & mask;
}

// -----------------------------------------------------------------------------
// elect_one (sm90 ELECT / PTX elect.sync): elect a single leader — the LOWEST
// active lane of the converged history-matched group — and return its lane id
// (uniform across the warp). The lifter derives ELECT's per-lane predicate as
// (myLane == leader) and writes the leader id to ELECT's register dest.
//
// Like vote_*/redux this is a REAL election over the gathered membership (not
// an unconditional "lane 0 is elected"): the x86 wrapper runs every warp lane
// concurrently, so electing all lanes would let each pass the typical
// `@!P EXIT` guard and race the leader-only store (ISA2a_elect_one). Gathering
// also makes divergent / partial warps elect the lowest *active* lane, matching
// HW elect.sync.
// -----------------------------------------------------------------------------
extern "C" uint32_t elect_one(uint32_t site)
{
    (void)site;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return (uint32_t)g_threadCtx.lane;  // sole lane leads
    // The leader is the LOWEST lane of the converged group = lane 0. elect.sync
    // requires convergence (it is `.sync`), so every lane that executes this ELECT
    // is a member and the lowest is deterministically lane 0 -- no cross-lane
    // communication is needed to know that.
    //
    // We must NOT recover this via a blocking warp_group_gather rendezvous: in a
    // warp-SPECIALIZED producer loop (FlashAttention-3) the elected leader (lane 0)
    // does divergent downstream work after the ELECT (issues the TMA, arrives on an
    // mbarrier, then waits on the consumer's empty-barrier) while lanes 1..31 skip
    // it and loop back to the next ELECT. A gather rendezvous makes lanes 1..31
    // BLOCK at that next ELECT waiting for lane 0, which is itself blocked
    // downstream on the consumer -> a cross-warpgroup deadlock that cannot exist on
    // lockstep HW (where all 32 lanes retire the ELECT in one cycle). Returning the
    // constant leader (lane 0) lets the non-leader lanes correctly evaluate their
    // (myLane==leader) predicate as false and proceed; lane 0 runs the leader-only
    // work whenever it arrives. Electing lane 0 (not "all lanes") still keeps the
    // leader-only store single-writer (the ISA2a_elect_one requirement).
    return 0;
}

// -----------------------------------------------------------------------------
// Hopper mbarrier (SYNCS.EXCH=init / .ARRIVE=arrive / .PHASECHK.TRYWAIT=wait).
//
// UNCONDITIONAL + general (no env toggle): the lifter emits these calls whenever
// a kernel uses mbarrier ops, so warp-specialized producer/consumer pipelines
// (FlashAttention-3) get a REAL barrier — the consumer actually blocks on the
// producer's arrivals. Under the wrapper's concurrent per-thread execution that
// happens-before is otherwise absent (a consumer warp with lower tids runs first
// and reads shared memory before the producer fills it). A kernel that uses no
// mbarrier ops emits no calls and is unaffected.
//
// State is keyed by (CTA, smem byte offset): each CTA owns its own shared
// barriers, and concurrent CTAs must not alias. mbarrier.init fully resets a
// key's state, so re-launches and phase reuse are handled without an explicit
// global reset.
struct MbarState {
    std::mutex m;
    std::condition_variable cv;
    int       expected = 1;   // arrivals needed to flip a phase
    int       arrived  = 0;   // arrivals seen in the current phase
    long long tx       = 0;   // outstanding transaction bytes (TMA); flip needs tx<=0
    int       phase    = 0;   // current (not-yet-complete) phase parity
    bool      inited   = false;
};
static std::unordered_map<uint64_t, MbarState> g_mbar;
static std::mutex g_mbar_mtx;

static inline uint64_t mbar_key(uint32_t off) {
    return ((uint64_t)(uint32_t)g_threadCtx.cta << 32) | off;
}
static MbarState* mbar_get(uint32_t off) {
    std::lock_guard<std::mutex> lk(g_mbar_mtx);
    return &g_mbar[mbar_key(off)];       // node-based map: references stay valid
}
// Hopper mbarrier.init writes a packed value of the form (0x200000 - 2*count) in
// its low field; recover the expected arrival count from it. A plain small count
// (when ptxas passes it directly) is used as-is.
static inline int mbar_count_from_init(uint64_t v) {
    uint32_t lo = (uint32_t)(v & 0x3fffff);
    if (lo == 0) return 1;
    if (lo <= 0x100000) return (int)lo;          // direct small count
    int cnt = (int)((0x200000 - lo) >> 1);       // packed pending-count form
    return cnt > 0 ? cnt : 1;
}
static inline bool mbar_dbg() {
    static const bool e = std::getenv("SLIFTER_MBAR_DEBUG") != nullptr; return e;
}
// DEBUG (gated): unified data-lifetime tracker for the FA3 bwd LSE/dPsum tile.
// A single monotonic execution counter (g_lse_seq) stamps EVERY event so the
// producer UBLKCP writes, any STS into the region, and the consumer LDS reads
// appear in one totally-ordered stream. For each 8-byte slot in the LSE/dPsum
// window we remember the seq + value of the LAST write; the LDS dump prints both
// "value I read now" and "last writer (seq, val)" so a stale/clobbered read is
// unambiguous. Region tracked: 0x2e000 .. 0x2e6c0 (the 4 ring tiles). cta 0 only.
// All gated on SLIFTER_LSE_TRACE at lift+run time -> production byte-identical.
// (constants + lse_trace_record_write defined above, before utma_load.)
extern "C" void slifter_ublkcp_dump(uint32_t dst_off, const void* src, int64_t bytes) {
    if (!std::getenv("SLIFTER_UBLKCP_DUMP") && !lse_trace_on()) return;
    float v0 = 0.f; if (src) std::memcpy(&v0, src, sizeof(float));
    if (lse_trace_on() && g_threadCtx.cta == 0) {
        long seq = g_lse_seq.fetch_add(1);
        bool hit = (dst_off < LSE_TRACE_HI) && (dst_off + (uint32_t)bytes > LSE_TRACE_LO);
        if (hit) {
            std::fprintf(stderr, "[seq %ld] [ublkcp] cta=0 tid=%d dst=0x%x bytes=%lld src[0]=%g\n",
                         seq, g_threadCtx.tid, dst_off, (long long)bytes, v0);
            lse_trace_record_write(dst_off, (uint32_t)bytes,
                                   reinterpret_cast<const float*>(src), 'u', seq);
        }
    } else if (std::getenv("SLIFTER_UBLKCP_DUMP")) {
        std::fprintf(stderr, "[ublkcp] cta=%d tid=%d dst=0x%x bytes=%lld src[0]=%g\n",
                     g_threadCtx.cta, g_threadCtx.tid, dst_off, (long long)bytes, v0);
    }
}
// DEBUG (gated): the consumer-side counterpart. Prints the EFFECTIVE shared-memory
// offset an LDS in the LSE/dPsum region reads, the value it reads NOW, and (under
// SLIFTER_LSE_TRACE) the seq + value of the LAST writer to that 8-byte slot.
extern "C" void slifter_lds_dump(uint32_t eff_off, uint32_t imm_off) {
    bool trace = lse_trace_on();
    if (!std::getenv("SLIFTER_LDS_DUMP") && !trace) return;
    if (g_threadCtx.cta != 0) return;           // count only cta 0, so the limit isn't
    static std::atomic<int> n{0};               // exhausted by other CTAs first
    if (n.fetch_add(1) < 60) {
        float v = 0.f;                          // peek the value the LDS is about to read
        std::memcpy(&v, reinterpret_cast<const uint8_t*>(shared_mem) + eff_off, sizeof(float));
        if (trace) {
            long seq = g_lse_seq.fetch_add(1);
            long wseq = -1; float wval = 0.f; char wk = '?';
            if (eff_off >= LSE_TRACE_LO && eff_off < LSE_TRACE_HI) {
                const LseSlotInfo& wi = g_lse_last_write[(eff_off - LSE_TRACE_LO) / 8u];
                wseq = wi.seq; wval = wi.val; wk = wi.kind ? wi.kind : '0';
            }
            std::fprintf(stderr, "[seq %ld] [lds] cta=0 tid=%d eff=0x%x imm=0x%x readval=%g  "
                         "lastwrite(seq=%ld kind=%c val=%g)\n",
                         seq, g_threadCtx.tid, eff_off, imm_off, v, wseq, wk, wval);
        } else {
            std::fprintf(stderr, "[lds] cta=%d tid=%d eff_off=0x%x imm=0x%x val=%g\n",
                         g_threadCtx.cta, g_threadCtx.tid, eff_off, imm_off, v);
        }
    }
}
// DEBUG (gated SLIFTER_LSE_TRACE): record an STS/STSM into the tracked LSE window
// in the unified seq stream, so a stray scalar store that clobbers the LSE tile is
// caught alongside the bulk copies. The lifted call is only emitted under the env
// at lift time -> production byte-identical. `nbytes` is the store width.
extern "C" void slifter_sts_dump(uint32_t eff_off, const void* valptr, uint32_t nbytes) {
    if (!lse_trace_on()) return;
    if (g_threadCtx.cta != 0) return;
    if (eff_off >= LSE_TRACE_HI || eff_off + nbytes <= LSE_TRACE_LO) return;  // outside window
    long seq = g_lse_seq.fetch_add(1);
    float v0 = 0.f; if (valptr) std::memcpy(&v0, valptr, sizeof(float) <= nbytes ? sizeof(float) : nbytes);
    std::fprintf(stderr, "[seq %ld] [sts] cta=0 tid=%d eff=0x%x bytes=%u val0=%g\n",
                 seq, g_threadCtx.tid, eff_off, nbytes, v0);
    lse_trace_record_write(eff_off, nbytes, &v0, 's', seq);
}
extern "C" void slifter_mbar_init(uint32_t off, uint64_t initval) {
    MbarState* s = mbar_get(off);
    std::lock_guard<std::mutex> lk(s->m);
    s->expected = mbar_count_from_init(initval);
    s->arrived = 0; s->tx = 0; s->phase = 0; s->inited = true;
    if (mbar_dbg()) std::fprintf(stderr, "[mbar] init cta=%d off=0x%x iv=0x%llx expected=%d\n",
                                 g_threadCtx.cta, off, (unsigned long long)initval, s->expected);
    s->cv.notify_all();
}
static inline void mbar_maybe_flip(MbarState* s) {
    if (s->arrived >= s->expected && s->tx <= 0) {
        s->arrived -= s->expected; s->phase ^= 1; s->cv.notify_all();
    }
}
extern "C" void slifter_mbar_arrive(uint32_t off, int64_t tx_add) {
    // Producer-side release: publish prior shared-memory writes before the phase
    // flip becomes visible to a consumer's try_wait.
    std::atomic_thread_fence(std::memory_order_release);
    MbarState* s = mbar_get(off);
    std::lock_guard<std::mutex> lk(s->m);
    s->arrived += 1; s->tx += tx_add; mbar_maybe_flip(s);
    if (mbar_dbg()) std::fprintf(stderr, "[mbar] arrive cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
                                 g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
}

// cp.async-completion arrive — `ARRIVES.LDGSTSBAR.TRANSCNT [bar]`.
//
// On Hopper this arms the barrier's transaction-completion tracking for the
// outstanding cp.async (LDGSTS) bytes; it does NOT itself contribute a plain
// arrival count. Each such ARRIVES is PAIRED, on the SAME barrier offset, with a
// `SYNCS.ARRIVE.TRANS64.A1T0 [bar], RZ` (the +1 count). The FA3 forward gather
// producers (packgqa / split / paged / paged_split) emit BOTH per stage.
//
// On the serialized-warp (fiber) backend the cp.async is lowered to a SYNCHRONOUS
// LDGSTS that has already published its bytes to smem by the time this arrive
// runs, so the transaction is already satisfied — there is no outstanding tx to
// track. The paired SYNCS.ARRIVE supplies the count. If ARRIVES ALSO incremented
// the count, the full (data-ready) barrier would receive 2 arrivals per producer
// thread and flip after only HALF the producer warpgroup has gathered — the
// proven paged_split premature-flip (consumers read WG2's Q as zero, corr 0.977).
//
// So ARRIVES here is a release fence + a NOP on the count: it publishes prior
// smem writes (the gather) but defers the phase flip to the paired SYNCS.ARRIVE.
// Gated at lift time (lift_arrives routes here only for the gather producers, via
// self._has_ldgsts) so TMA-only kernels (no ARRIVES) stay byte-identical.
extern "C" void slifter_mbar_ldgsts_arrive(uint32_t off) {
    std::atomic_thread_fence(std::memory_order_release);
    if (mbar_dbg()) {
        MbarState* s = mbar_get(off);
        std::lock_guard<std::mutex> lk(s->m);
        std::fprintf(stderr, "[mbar] ldgsts_arrive(nocount) cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
                     g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
    }
    // No count increment, no flip: the paired SYNCS.ARRIVE.A1T0 provides the count.
}

// ---------------------------------------------------------------------------
// Expect-tx (transaction-count) arrive — `mbarrier.arrive.expect_tx`.
//
// FA3 producers issue `SYNCS.ARRIVE.TRANS64 [bar], R<bytes>` (a NON-RZ expect-tx
// operand) and THEN issue the bulk copies (UTMALDG/UBLKCP) that fill the tile.
// On hardware the barrier does NOT flip at the arrive: it flips only once the
// expected transaction bytes have landed (the TMA/cp.async.bulk completion).
//
// On the serialized-warp (fiber) backend a producer fiber runs straight-line
// from the arrive through its bulk copies with NO yield in between (the copies
// are lowered to SYNCHRONOUS memcpy/utma_load that execute in program order).
// So if the arrive flipped the phase immediately (as the plain count-arrive
// does), a consumer parked in try_wait would be released and read the smem tile
// BEFORE the producer fiber executes the copy — the long-flagged LSE/dPsum
// "reads zero" bug (P=exp2(...-0)=1.0, dV/dK/dQ garbage).
//
// Fix, general + deadlock-free: a deferred-flip arrive. It counts the arrival
// but does NOT flip; it records the barrier as "pending" for THIS fiber. The
// flip is performed by slifter_mbar_flush_pending(), which the fiber scheduler
// calls at every yield point (warp_fiber_yield) — i.e. the instant the producer
// fiber relinquishes control. By then its synchronous bulk copies have already
// executed, so the tile is published in smem before any consumer can observe
// the phase flip. Plain RZ-expect-tx arrives (ws_attn_pipe) keep using
// slifter_mbar_arrive and are byte-identical / unaffected. In the per-thread
// (non-fiber) backend there is no producer/consumer race on one OS thread, so
// the flush also runs eagerly there (flush is invoked, harmlessly, by the same
// hook only under fibers — see below).
// Fix, general + deadlock-free: a deferred-flip arrive that flips a barrier only
// AFTER its bulk copy actually runs. `arrive_tx` records the barrier pending +
// UNDELIVERED; each bulk copy (UBLKCP/UTMALDG) calls slifter_mbar_deliver_tx(),
// which marks the most-recent arrive's barrier delivered; the yield-flush flips
// only DELIVERED barriers. This matters because a producer fiber can YIELD between
// the expect-tx arrive and its UBLKCP (FA3 bwd LSE/dPsum): a plain flush-at-yield
// would flip the phase before the copy ran, releasing the consumer to read 0. The
// `stale` fallback flips a long-undelivered barrier anyway (avoids deadlock if an
// arrive has no tracked copy). g_recent_tx_off is safe across fibers because FA3
// has a single producer (TMA) warp -> only it ever sets it.
// Byte-accounting expect-tx FIFO. arrive_tx(off, bytes) records the barrier + the
// number of transaction bytes it expects (a FIFO entry). Each bulk copy
// (UBLKCP/UTMALDG) calls slifter_mbar_deliver_tx(bytes) reporting how many bytes it
// just published; those bytes retire the OLDEST outstanding barrier (the producer
// issues arrive(A),copies(A),arrive(B),copies(B)... so the copy stream retires the
// FIFO in order). When a barrier's bytes are fully delivered it flips THEN — only
// after its own copy ran, even if the producer yielded between the arrive and the
// copy, and even under multi-atom copies / pipelining (the failure modes the plain
// flush-at-yield and most-recent-arrive models both hit). Gated on _has_ublkcp so
// forward kernels (no expect-tx deferral) are byte-identical. A stale fallback in
// the yield-flush flips a long-undelivered head to avoid deadlock on any mismatch.
struct PendingTx { uint32_t off; int64_t remaining; int stale; };
static thread_local std::deque<PendingTx> g_mbar_pending_tx;

extern "C" void slifter_mbar_arrive_tx(uint32_t off, int64_t bytes) {
    std::atomic_thread_fence(std::memory_order_release);
    {
        MbarState* s = mbar_get(off);
        std::lock_guard<std::mutex> lk(s->m);
        s->arrived += 1;   // count the arrival; flip is deferred to byte delivery
        if (mbar_dbg()) std::fprintf(stderr,
            "[mbar] arrive_tx cta=%d off=0x%x tid=%d bytes=%lld -> %d/%d phase=%d\n",
            g_threadCtx.cta, off, g_threadCtx.tid, (long long)bytes, s->arrived, s->expected, s->phase);
    }
    g_mbar_pending_tx.push_back({off, bytes > 0 ? bytes : 1, 0});
}

// A bulk copy published `bytes` bytes: retire them against the oldest outstanding
// expect-tx barrier; flip it once fully delivered. No-op when nothing is pending
// (every forward kernel) -> byte-identical.
extern "C" void slifter_mbar_deliver_tx(int64_t bytes) {
    std::atomic_thread_fence(std::memory_order_release);
    if (getenv("SLIFTER_DELIVER_DBG") && g_threadCtx.cta == 0) {
        static std::atomic<int> nd{0};
        if (nd.fetch_add(1) < 30)
            std::fprintf(stderr, "[deliver] cta=0 tid=%d bytes=%lld front=%s%x rem=%lld depth=%zu\n",
                g_threadCtx.tid, (long long)bytes,
                g_mbar_pending_tx.empty() ? "EMPTY" : "0x",
                g_mbar_pending_tx.empty() ? 0u : g_mbar_pending_tx.front().off,
                g_mbar_pending_tx.empty() ? 0LL : (long long)g_mbar_pending_tx.front().remaining,
                g_mbar_pending_tx.size());
    }
    while (bytes > 0 && !g_mbar_pending_tx.empty()) {
        PendingTx& f = g_mbar_pending_tx.front();
        f.remaining -= bytes;
        if (f.remaining > 0) return;          // barrier still expecting more bytes
        // Fully delivered: flip it now (its copy has run) and carry any excess on.
        bytes = -f.remaining;
        uint32_t off = f.off;
        g_mbar_pending_tx.pop_front();
        MbarState* s = mbar_get(off);
        std::lock_guard<std::mutex> lk(s->m);
        mbar_maybe_flip(s);
        if (mbar_dbg()) std::fprintf(stderr,
            "[mbar] deliver_tx(flip) cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
            g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
    }
}

// ---------------------------------------------------------------------------
// Count-deferred-flip arrive — `mbarrier.arrive` (plain count) whose phase flip
// must be deferred to the producer fiber's next yield.
//
// FA3 fp8 (e4m3/e5m2) fwd is warp-specialized: warpgroup 0 is a Vt-TRANSPOSE
// producer (it reads V from TMA smem and re-publishes it MN->K transposed via a
// warp-collective STSM into a 2-stage smem ring), and warpgroups 1/2 are PV
// consumers that read Vt from that ring. The producer signals "this stage's Vt
// is published" with a PLAIN COUNT arrive `SYNCS.ARRIVE.TRANS64.A1T0 [bar], RZ`
// (RZ = no expect-tx bytes; the publish is a register->smem STSM, not a bulk
// copy, so the byte-accounting expect-tx FIFO above does NOT apply).
//
// On HW the STSM and the arrive are ordered by the warp's program order + the
// consumer's mbarrier wait. On the serialized-warp (fiber) backend the producer
// fiber runs straight-line from the arrive through its synchronous STSM with no
// yield in between, BUT — exactly as for expect-tx — if the arrive flipped the
// phase eagerly (slifter_mbar_arrive) a consumer parked in try_wait would be
// released and read the Vt stage BEFORE the producer fiber executes the STSM.
// That is the proven e4m3 PV-drop bug: the partial last K-tile's PV reads zero
// V bytes (O = 224/256 * v_descale). fp16/bf16 fwd have NO Vt transpose (V is
// read MN-major straight from TMA smem) so they never take this path.
//
// Fix: a count deferred-flip arrive. It counts the arrival (so `expected` is
// satisfied) but does NOT flip; it records the barrier as count-pending for THIS
// fiber. slifter_mbar_flush_pending() (called at every warp_fiber_yield and at
// fiber exit) flips it — by then the producer fiber's STSM has published the Vt
// stage to smem, so a consumer may finally observe the flip and read real data.
// This mirrors the expect-tx deferred-flip, but with a COUNT predicate (no byte
// accounting) since the publish is a warp-collective store, not a bulk copy.
// Gated narrowly at lift time (only the fp8 .A1T0 non-RED transpose-producer
// arrive routes here) so fp16/bf16/bwd and all other arrives are byte-identical.
struct PendingCount { uint32_t off; };
static thread_local std::deque<PendingCount> g_mbar_pending_count;

extern "C" void slifter_mbar_arrive_deferred(uint32_t off) {
    std::atomic_thread_fence(std::memory_order_release);
    {
        MbarState* s = mbar_get(off);
        std::lock_guard<std::mutex> lk(s->m);
        s->arrived += 1;   // count the arrival; flip deferred to the next yield
        if (mbar_dbg()) std::fprintf(stderr,
            "[mbar] arrive_deferred cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
            g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
    }
    g_mbar_pending_count.push_back({off});
}

// Yield-time publish + safety net. (1) Flip every count-deferred arrive armed by
// this fiber: by the yield point its STSM publish has run, so the consumer may
// now observe the flip. (2) Flip an expect-tx head barrier that has gone many
// yields without its bytes arriving (guards against any arrive/copy byte
// mismatch causing deadlock). No-op when nothing is pending -> byte-identical.
extern "C" void slifter_mbar_flush_pending() {
    if (!g_mbar_pending_count.empty()) {
        std::atomic_thread_fence(std::memory_order_release);
        while (!g_mbar_pending_count.empty()) {
            uint32_t off = g_mbar_pending_count.front().off;
            g_mbar_pending_count.pop_front();
            MbarState* s = mbar_get(off);
            std::lock_guard<std::mutex> lk(s->m);
            mbar_maybe_flip(s);   // flips only once arrived>=expected (count-arrive)
            if (mbar_dbg()) std::fprintf(stderr,
                "[mbar] flush_count(flip) cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
                g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
        }
    }
    if (g_mbar_pending_tx.empty()) return;
    std::atomic_thread_fence(std::memory_order_release);
    while (!g_mbar_pending_tx.empty() && ++g_mbar_pending_tx.front().stale > 32) {
        uint32_t off = g_mbar_pending_tx.front().off;
        g_mbar_pending_tx.pop_front();
        MbarState* s = mbar_get(off);
        std::lock_guard<std::mutex> lk(s->m);
        mbar_maybe_flip(s);
        if (mbar_dbg()) std::fprintf(stderr,
            "[mbar] flush_tx(stale-flip) cta=%d off=0x%x tid=%d -> %d/%d phase=%d\n",
            g_threadCtx.cta, off, g_threadCtx.tid, s->arrived, s->expected, s->phase);
    }
}
// TMA bulk-copy completion: decrement the barrier's outstanding tx bytes.
extern "C" void slifter_mbar_complete_tx(uint32_t off, int64_t bytes) {
    MbarState* s = mbar_get(off);
    std::lock_guard<std::mutex> lk(s->m);
    s->tx -= bytes; mbar_maybe_flip(s);
}
// Returns 1 (ready) iff the barrier has advanced past `parity`. Non-blocking:
// the lifted kernel spins (try_wait + branch); we yield when not ready so the
// producer that must arrive can run (avoids livelock on limited cores). An
// un-init'd barrier spins (not ready) rather than falsely reporting ready —
// otherwise a consumer could read a stage before the producer fills it.
extern "C" int slifter_mbar_try_wait(uint32_t off, uint32_t parity) {
    MbarState* s = mbar_get(off);
    // Hopper encodes phase parity in bit 31 (phase 1 = 0x80000000, phase 0 = 0);
    // (!=0) extracts it robustly regardless of bit-0/bit-31 form.
    int pbit = (parity != 0u) ? 1 : 0;
    // Serialized-warp fiber path: single OS thread, so don't sleep on the cv
    // (that freezes the producer lane too). If ready, return 1; else YIELD to let
    // the producer lane run + arrive, then re-check. The lifted retry loop
    // re-calls try_wait, so one yield per call advances the schedule.
    if (warp_fiber_active()) {
        if (mbar_dbg() && off == 0x38810u && g_threadCtx.cta == 0) {  // LSE barrier
            static std::atomic<int> tw{0};
            if (tw.fetch_add(1) < 10)
                std::fprintf(stderr, "[trywait] cta=0 tid=%d off=0x%x phase=%d pbit=%d parity=0x%x inited=%d -> ready=%d\n",
                             g_threadCtx.tid, off, s->phase, pbit, parity, (int)s->inited,
                             (int)(s->inited && s->phase != pbit));
        }
        if (s->inited && s->phase != pbit) return 1;
        g_warp_yield();
        if (s->inited && s->phase != pbit) return 1;
        if (mbar_dbg()) {   // localize a stuck fiber's wait (print once per tid/off/phase)
            static std::mutex fwm; static std::set<uint64_t> fseen;
            uint64_t k = ((uint64_t)(uint32_t)g_threadCtx.tid << 40) ^ ((uint64_t)off << 8) ^ (uint64_t)(pbit & 1);
            std::lock_guard<std::mutex> fl(fwm);
            if (fseen.insert(k).second)
                std::fprintf(stderr, "[mbar] FIBER-WAIT cta=%d off=0x%x tid=%d wg=%d want!=%d cur=%d arrived=%d/%d inited=%d\n",
                             g_threadCtx.cta, off, g_threadCtx.tid, g_threadCtx.tid/128, pbit,
                             s->phase, s->arrived, s->expected, (int)s->inited);
        }
        return 0;
    }
    {
        std::unique_lock<std::mutex> lk(s->m);
        // When not ready, SLEEP on the cv (woken by arrive/init) up to a short
        // timeout instead of busy-yielding. At warp-group scale (block=384 ->
        // hundreds of threads spinning) busy-yield is a thundering herd that
        // starves the producer that must arrive -> the FA3 "perf wall" (100s+).
        // The timeout keeps a never-completing barrier from blocking forever
        // (the lifted kernel re-polls). Un-init'd waits (not ready), never
        // falsely ready (else a consumer reads a stage before the producer fills).
        if (!(s->inited && s->phase != pbit))
            s->cv.wait_for(lk, std::chrono::microseconds(200),
                           [&]{ return s->inited && s->phase != pbit; });
        if (!(s->inited && s->phase != pbit)) {
            if (mbar_dbg()) {
                // Pure-logging: report each (tid,off,parity) wait once so we can
                // see which barrier a stuck consumer polls vs what producers arm.
                static std::mutex wm; static std::set<uint64_t> seen;
                uint64_t k = ((uint64_t)(uint32_t)g_threadCtx.tid << 40) ^
                             ((uint64_t)off << 8) ^ (uint64_t)(pbit & 1);
                std::lock_guard<std::mutex> wl(wm);
                if (seen.insert(k).second)
                    std::fprintf(stderr, "[mbar] WAIT cta=%d off=0x%x tid=%d wg=%d "
                                 "want_phase!=%d inited=%d cur_phase=%d arrived=%d/%d\n",
                                 g_threadCtx.cta, off, g_threadCtx.tid,
                                 g_threadCtx.tid/128, pbit, (int)s->inited,
                                 s->phase, s->arrived, s->expected);
            }
            return 0;   // still not ready: kernel re-polls (sleeps again)
        }
    }
    // Consumer-side acquire: the producer's shared writes are now visible to the
    // smem load the lifted kernel performs immediately after this returns ready.
    std::atomic_thread_fence(std::memory_order_acquire);
    return 1;
}

// -----------------------------------------------------------------------------
// Named / partial barriers — BAR.SYNC / BAR.ARV <id>, <count>.
//   void named_barrier_sync(uint32_t id, uint32_t count)    // arrive + wait
//   void named_barrier_arrive(uint32_t id, uint32_t count)  // arrive, no wait
//
// A warp-specialized kernel (FA3) synchronizes a SUBSET of the CTA — e.g. its
// two consumer warpgroups (256 threads) — at a named barrier, distinct from the
// full-CTA __syncthreads. On the PER-THREAD x86 backend there are no warpgroups
// (each CUDA thread is its own OS thread; ordering comes from real barriers /
// mbarriers), so these are NOPs — keeping that path byte-identical. On the WARP
// (fiber) backend they are real partial barriers: a generation barrier per
// (CTA, id) that yields the fiber until `count` participants have arrived. The
// fiber scheduler runs one OS thread per CTA cooperatively, so the counter
// updates between yields are atomic and need no lock.
static constexpr int  SLIFTER_MAX_NAMED_BAR = 16;   // BAR id is 4 bits
struct NamedBarCtx { int arrived = 0; int generation = 0; };
static NamedBarCtx*    g_namedBarCtx = nullptr;      // [numCTA * 16], cta*16 + (id&15)

static inline bool nbar_dbg() { static int v = getenv("SLIFTER_NBAR_DEBUG") ? 1 : 0; return v; }

extern "C" void named_barrier_sync(uint32_t id, uint32_t count) {
    if (!warp_fiber_active() || !g_namedBarCtx) return;          // per-thread: NOP
    NamedBarCtx& ctx = g_namedBarCtx[(size_t)g_threadCtx.cta * SLIFTER_MAX_NAMED_BAR
                                     + (id & (SLIFTER_MAX_NAMED_BAR - 1))];
    const int myGen = ctx.generation;
    int a = ++ctx.arrived;
    if (nbar_dbg() && a == 1)
        std::fprintf(stderr, "[nbar] SYNC enter id=%u count=%u tid=%d (first arriver)\n", id, count, g_threadCtx.tid);
    if (a >= (int)count) {
        if (nbar_dbg()) std::fprintf(stderr, "[nbar] SYNC COMPLETE id=%u count=%u (tid=%d)\n", id, count, g_threadCtx.tid);
        ctx.arrived = 0; ++ctx.generation;
    } else {
        int spins = 0;
        while (ctx.generation == myGen) {
            g_warp_yield();                                       // fiber: yield, peers run
            if (nbar_dbg() && ++spins == 200000) {
                std::fprintf(stderr, "[nbar] SYNC STUCK id=%u count=%u arrived=%d tid=%d\n",
                             id, count, ctx.arrived, g_threadCtx.tid);
                spins = 0;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// RS-form WGMMA A operand (FlashAttention PV GEMM: A=P held in REGISTERS).
// On the GPU `wgmma` reads the A fragment from the whole warpgroup register file;
// on x86 fibers/threads don't share registers, so each thread SPILLS its 8-fp16
// A fragment to a per-warpgroup shared scratch, warpgroup-barriers, then the
// WGMMA decode reads A[m,k] from scratch. A-fragment layout (cute MMA_Traits
// ALayout, probed exact): the 4 f16x2 regs r0..r3 of thread tid map to
//   reg r -> lo=(m_base+(r&1)*8, k_base+(r>>1)*8), hi=(same m, k+1)
//   m_base = t1+16*t2 (t1=(tid>>2)&7, t2=(tid>>5)&3), k_base = 2*(tid&3).
// A is the 64x16 P slice; scratch is [64*16] fp16 per warpgroup.
static thread_local uint16_t g_rs_a_scratch[16][64 * 16];
static NamedBarCtx*          g_rsBarCtx = nullptr;   // [numCTA*16], per (cta, warpgroup)

static inline void rs_wg_barrier(int wg) {
    if (!warp_fiber_active() || !g_rsBarCtx) return;   // per-thread: NOP (no reg sharing path)
    NamedBarCtx& ctx = g_rsBarCtx[(size_t)g_threadCtx.cta * SLIFTER_MAX_NAMED_BAR + (wg & 15)];
    const int myGen = ctx.generation;
    if (++ctx.arrived >= 128) { ctx.arrived = 0; ++ctx.generation; }
    else { while (ctx.generation == myGen) g_warp_yield(); }
}

// Spill this thread's 4 f16x2 A-fragment regs to the warpgroup scratch. Brackets
// the write with two warpgroup barriers: the first ensures the PREVIOUS HGMMA's
// reads finished before we overwrite; the second publishes all fragments before
// the decode reads them.
extern "C" void slifter_rs_a_spill(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    // Hot path (called once per RS HGMMA per thread): read the debug env vars
    // ONCE (thread-safe magic statics), not per call.
    static const bool tma_dbg = std::getenv("SLIFTER_TMA_DEBUG") != nullptr;
    static const bool p_dump  = std::getenv("SLIFTER_P_DUMP") != nullptr;
    const int tid = g_threadCtx.tid;
    const int wg  = (tid / 128) & 15;
    if (tma_dbg) {
        static std::atomic<int> ne{0};
        int c = ne.fetch_add(1);
        if (c < 4) std::fprintf(stderr, "[rs_a_spill] ENTER #%d tid=%d wg=%d lane=%d\n", c, tid, wg, g_threadCtx.lane);
    }
    rs_wg_barrier(wg);
    const int t0 = tid & 3, t1 = (tid >> 2) & 7, t2 = (tid >> 5) & 3;
    const int m_base = t1 + 16 * t2, k_base = 2 * t0;
    const uint32_t regs[4] = {r0, r1, r2, r3};
    for (int r = 0; r < 4; ++r) {
        const int m = m_base + (r & 1) * 8;
        const int k = k_base + (r >> 1) * 8;
        // f16x2 packs the two K values LOW half = k, HIGH half = k+1. (A race-free
        // P-dump compared to a CPU exp(scale*Q*K^T - max) reference at the CORRECT
        // key base showed the previous hi->k packing put key(k+1) at scratch[k] —
        // i.e. scratch held P[key ^ 1], so the PV contracted P[key^1].V[key]
        // (mismatched keys) for a seq-correlated ~0.014 error. lo->k fixes it.)
        g_rs_a_scratch[wg][m * 16 + k]     = (uint16_t)(regs[r] & 0xffff);  // lo -> k
        g_rs_a_scratch[wg][m * 16 + k + 1] = (uint16_t)(regs[r] >> 16);     // hi -> k+1
    }
    if (tma_dbg) {
        static std::atomic<int> nps{0};
        if (g_threadCtx.lane == 0 && nps.fetch_add(1) < 8) {
            auto h2f = [](uint16_t b)->float{
                uint32_t s=(b>>15)&1u, e=(b>>10)&0x1fu, m=b&0x3ffu, f;
                if(e==0){ if(m==0){f=s<<31;} else {int ee=-1; do{ee++;m<<=1;}while(!(m&0x400u)); m&=0x3ffu; f=(s<<31)|((127-15-ee)<<23)|(m<<13);} }
                else if(e==0x1f){ f=(s<<31)|(0xffu<<23)|(m<<13); }
                else { f=(s<<31)|((e-15+127)<<23)|(m<<13); }
                float r; std::memcpy(&r,&f,4); return r; };
            std::fprintf(stderr, "[rs_a P] tid=%d wg=%d P[%d,%d..]= %.4f %.4f %.4f %.4f (m_base=%d k_base=%d)\n",
                         g_threadCtx.tid, wg, m_base, k_base,
                         h2f(regs[0]&0xffff), h2f(regs[0]>>16), h2f(regs[1]&0xffff), h2f(regs[1]>>16),
                         m_base, k_base);
        }
    }
    rs_wg_barrier(wg);
    // DEBUG (gated, so the extra barrier only exists in debug runs -> no production
    // behavior change): dump wg1's first-k-block P once, race-free. Lane 0 dumps,
    // then ALL threads hit the gated barrier so no peer can begin the next k-block
    // spill (overwriting the scratch) until the dump completes.
    if (p_dump) {
        if (wg == 1 && g_threadCtx.lane == 0 && g_threadCtx.cta == 0) {
            static std::atomic<int> pd{0};
            int c = pd.fetch_add(1);
            if (c < 64) {   // dump first 64 spills (covers both K-tiles' blocks) for full P check
                char nm[256];
                snprintf(nm, sizeof(nm), "p_spill%d.bin", c);
                FILE* f = fopen(slifter_debug_path(nm).c_str(), "wb");
                if (f) { fwrite(g_rs_a_scratch[wg], 2, 64 * 16, f); fclose(f); }
            }
        }
        rs_wg_barrier(wg);  // gated: peers wait for lane 0's dump before reusing scratch
    }
}

extern "C" uint32_t slifter_rs_a_read(int32_t m, int32_t k) {
    const int wg = (g_threadCtx.tid / 128) & 15;
    uint16_t bits = g_rs_a_scratch[wg][m * 16 + k];
    // DEBUG (gated SLIFTER_RSA_DUMP): record (tid, wg, m=key, k=query, P) for cta0,
    // wg1 only, capped, BUFFERED (no per-call fflush) to pin whether the RS-A read of
    // P varies across the output key (m). Off in production -> byte-identical.
    // Hoisted: this is the hottest runtime helper (per A-element per HGMMA).
    static const bool rsa_dump = std::getenv("SLIFTER_RSA_DUMP") != nullptr;
    if (rsa_dump && g_threadCtx.cta == 0 && wg == 1) {
        static std::atomic<int> nrec{0};
        int c = nrec.fetch_add(1);
        if (c < 200000) {
            static FILE* f = nullptr; static std::mutex mtx;
            std::lock_guard<std::mutex> lk(mtx);
            if (!f) f = fopen(slifter_debug_path("rsa_dump.bin").c_str(), "wb");
            if (f) {
                auto h2f=[](uint16_t b)->float{ uint32_t s=(b>>15)&1u,e=(b>>10)&0x1fu,mm=b&0x3ffu,fb;
                    if(e==0){ if(mm==0){fb=s<<31;} else {int ee=-1; do{ee++;mm<<=1;}while(!(mm&0x400u)); mm&=0x3ffu; fb=(s<<31)|((127-15-ee)<<23)|(mm<<13);} }
                    else if(e==0x1f){ fb=(s<<31)|(0xffu<<23)|(mm<<13);} else {fb=(s<<31)|((e-15+127)<<23)|(mm<<13);}
                    float r; std::memcpy(&r,&fb,4); return r; };
                float pv = h2f(bits);
                int32_t rec[4] = {g_threadCtx.tid, wg, m, k};
                fwrite(rec, 4, 4, f); fwrite(&pv, 4, 1, f);
            }
        }
    }
    return (uint32_t)bits;   // fp16 bits, zero-extended
}

// -----------------------------------------------------------------------------
// FP8 (e4m3) RS-form WGMMA A operand (FA3 FP8 PV GEMM: A=P held in registers).
// QGMMA.64x128x32 packs K=32 e4m3 bytes per MMA; the A fragment is 16 e4m3
// bytes/thread = 4 i32 regs (4 bytes each).
//
// CRITICAL fp8-only layout fact (root cause of the e4m3 real-V O scramble): P is
// PRODUCED by the QK QGMMA as its D-fragment, F2FP-packed into these 4 regs, then
// fed as the PV QGMMA's RS A-operand. For fp16/bf16 the HGMMA QK D-fragment == the
// HGMMA PV RS A-fragment, so a single (m,k) formula serves both (see
// slifter_rs_a_spill). For fp8 the QGMMA QK D-frag (F2FP-packed) DIFFERS from the
// QGMMA PV RS A-frag, so the register BYTES this spill receives are NOT in the PV
// A-frag (m,k) order the old code assumed — they must be stored at the (m,key) the
// F2FP-packed QK D-frag value actually holds. Any thread then reads
// scratch[m*32 + key] correctly (read unchanged).
//
// The (reg r, byte bi) -> (m, key) map was recovered EMPIRICALLY (no F2FP-order
// guess): the raw spill inputs were dumped under the real attention and each e4m3
// byte matched against true unnorm-P[query,key]*256; the closed form below scored
// corr 0.9995 vs true P across all 7 key blocks; the brute-force winner was
// unique — next best 0.58). It composes the F2FP packing order with the QK D-frag
// register layout:
//   m   = m_base + (r&1)*8                      (reg bit0 -> +8 query rows)
//   key = 2*t0 + (bi&1) + ((bi>>1)&1)*8 + ((r>>1)&1)*16
//   m_base = t1+16*t2 (t1=(tid>>2)&7, t2=(tid>>5)&3), t0=tid&3.
// (Contrast the OLD wrong PV-A-frag formula: m_off=(r&1)*8 [same!], but
//  k=4*t0 + bi + ((r>>1)&1)*16 — wrong key base 4*t0 and wrong byte->key split.)
// Scratch is the 64x32 P slice, one e4m3 byte per (m,key), per warpgroup.
static thread_local uint8_t g_rs_a_scratch8[16][64 * 32];

extern "C" void slifter_rs_a_spill_fp8(uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3) {
    const int tid = g_threadCtx.tid;
    const int wg  = (tid / 128) & 15;
    rs_wg_barrier(wg);
    const int t0 = tid & 3, t1 = (tid >> 2) & 7, t2 = (tid >> 5) & 3;
    const int m_base = t1 + 16 * t2;
    const uint32_t regs[4] = {r0, r1, r2, r3};
    for (int r = 0; r < 4; ++r) {
        for (int bi = 0; bi < 4; ++bi) {
            // QK D-fragment (m,key) the F2FP-packed byte actually holds (empirical).
            const int m   = m_base + (r & 1) * 8;
            const int key = 2 * t0 + (bi & 1) + ((bi >> 1) & 1) * 8 + ((r >> 1) & 1) * 16;
            g_rs_a_scratch8[wg][m * 32 + key] = (uint8_t)((regs[r] >> (8 * bi)) & 0xff);
        }
    }
    rs_wg_barrier(wg);
    static const bool p_dump = std::getenv("SLIFTER_P_DUMP") != nullptr;  // hoisted (hot path)
    if (p_dump) {
        if (wg == 1 && g_threadCtx.lane == 0 && g_threadCtx.cta == 0) {
            static std::atomic<int> pd{0};
            int c = pd.fetch_add(1);
            if (c < 64) {
                char nm[256];
                snprintf(nm, sizeof(nm), "p8_spill%d.bin", c);
                FILE* f = fopen(slifter_debug_path(nm).c_str(), "wb");
                if (f) { fwrite(g_rs_a_scratch8[wg], 1, 64 * 32, f); fclose(f); }
            }
        }
        rs_wg_barrier(wg);
    }
}

extern "C" uint32_t slifter_rs_a_read_fp8(int32_t m, int32_t k) {
    const int wg = (g_threadCtx.tid / 128) & 15;
    return (uint32_t)g_rs_a_scratch8[wg][m * 32 + k];   // e4m3 byte, zero-extended
}

// Diagnostic: dump the WGMMA decode's resolved A/B smem base offsets so we can
// compare b_base (V descriptor) against where utma_load WROTE V.
extern "C" void slifter_wgmma_dbg(int32_t a_base, int32_t b_base_lo, int32_t b_base_hi, int32_t is_rs) {
    if (!getenv("SLIFTER_TMA_DEBUG")) return;
    static std::atomic<int> nd{0};
    if (g_threadCtx.lane == 0 && nd.fetch_add(1) < 80)
        std::fprintf(stderr, "[wgmma] tid=%d wg=%d is_rs=%d a_base=%d b_lo=%d b_hi=%d\n",
                     g_threadCtx.tid, g_threadCtx.tid/128, is_rs, a_base, b_base_lo, b_base_hi);
}

// Diagnostic: dump the PV B-read smem byte offset (relative to shared_mem) for a
// single warpgroup+cta so it can be correlated against the in-kernel V-transpose
// STSM dst offsets ([stsm] dst_off). Gated by SLIFTER_PV_BOFF_DBG.
extern "C" void slifter_pv_boff_dbg(int32_t n_val, int32_t k_local, int32_t b_off) {
    if (!getenv("SLIFTER_PV_BOFF_DBG")) return;
    // Aggregate by hdim block: count, per block, how many reads see byte!=0 vs ==0.
    // Print a periodic summary (every 200000 reads) + the final picture, so we
    // capture LATE reads (after the V-transpose writes), not just iteration 0.
    static std::atomic<long> cnt0[4]{};   // zero-byte reads per hdim block
    static std::atomic<long> cntN[4]{};   // nonzero-byte reads per hdim block
    static std::atomic<long> total{0};
    if (g_threadCtx.cta == 0 && g_threadCtx.tid == 128) {
        uint8_t bv = ((const uint8_t*)shared_mem)[b_off];
        int blk = (n_val >> 5) & 3;
        if (bv) cntN[blk].fetch_add(1); else cnt0[blk].fetch_add(1);
        long t = total.fetch_add(1) + 1;
        if (t % 100000 == 0 || t == 22528) {
            std::fprintf(stderr, "[pvb-sum @%ld] blk0 nz=%ld z=%ld | blk1 nz=%ld z=%ld | blk2 nz=%ld z=%ld | blk3 nz=%ld z=%ld\n",
                t, cntN[0].load(),cnt0[0].load(), cntN[1].load(),cnt0[1].load(),
                cntN[2].load(),cnt0[2].load(), cntN[3].load(),cnt0[3].load());
        }
    }
}

// Diagnostic: dump the QK S fragment (scale*Q.K) for a single warpgroup+cta so it
// can be compared element-wise vs a CPU reference. Gated by SLIFTER_S_DUMP. Writes
// to s_dump.bin as records {tid,wg,m,n,s} for wg1,cta0.
extern "C" void slifter_s_dump(int32_t m, int32_t n, int32_t a_base, int32_t b_base, float s) {
    if (!getenv("SLIFTER_S_DUMP")) return;
    const int tid = g_threadCtx.tid;
    const int wg  = tid / 128;
    if (wg != 1 || g_threadCtx.cta != 0) return;
    static FILE* f = nullptr;
    static std::mutex mtx;
    static std::atomic<int> seq{0};   // global monotonic record index (no fflush)
    int s_idx = seq.fetch_add(1);
    if (s_idx >= 4000000) return;      // cap
    std::lock_guard<std::mutex> lk(mtx);
    if (!f) f = fopen(slifter_debug_path("s_dump.bin").c_str(), "wb");
    if (f) {
        int32_t rec[5] = {tid, m, n, a_base, b_base};
        fwrite(rec, 4, 5, f);   // tid, m, n, a_base, b_base
        fwrite(&s, 4, 1, f);    // running acc value
    }
}

// Diagnostic: dump the PV (RS-form) per-k operands A=P and B=V + b_base sub-block
// for ONE warpgroup+cta so the P/V key alignment can be checked vs CPU. Gated by
// SLIFTER_PVOP_DUMP. Records {tid,m,n,k_local,b_base,a,b} for wg1,cta0.
extern "C" void slifter_pvop_dump(int32_t m, int32_t n, int32_t k_local,
                                  int32_t b_base, float a, float b) {
    if (!getenv("SLIFTER_PVOP_DUMP")) return;
    const int tid = g_threadCtx.tid;
    if (tid / 128 != 1 || g_threadCtx.cta != 0) return;
    static FILE* f = nullptr;
    static std::mutex mtx;
    std::lock_guard<std::mutex> lk(mtx);
    if (!f) f = fopen(slifter_debug_path("pvop_dump.bin").c_str(), "wb");
    if (f) {
        int32_t rec[5] = {tid, m, n, k_local, b_base};
        fwrite(rec, 4, 5, f);
        fwrite(&a, 4, 1, f);
        fwrite(&b, 4, 1, f);
        fflush(f);
    }
}

// Diagnostic: dump the SS-form WGMMA operands (S/dP/dQ bwd GEMMs) per (m,n,k_local)
// with the resolved a_base/b_base smem byte offsets and the av/bv operand values, so
// we can pin whether the swapAB SS read lands on the K/Q smem the TMA wrote (suspect
// b: bases wrong) or reads zero (suspect c: data-not-there ordering). Records
// {tid,m,n,k_local,a_base,b_base,av,bv} for wg1,cta0. Gated by SLIFTER_SOP_DUMP.
extern "C" void slifter_sop_dump(int32_t m, int32_t n, int32_t k_local,
                                 int32_t a_base, int32_t b_base, float av, float bv) {
    if (!getenv("SLIFTER_SOP_DUMP") && !getenv("SLIFTER_SOP_DUMP_DQ")) return;
    const int tid = g_threadCtx.tid;
    if (tid / 128 != 1 || g_threadCtx.cta != 0) return;
    static FILE* f = nullptr;
    static std::mutex mtx;
    static long ncap = 0;
    std::lock_guard<std::mutex> lk(mtx);
    // Cap total records so a heavy dump can't run the process out of resources
    // before the kernel finishes (the truncation otherwise drops late GEMMs).
    if (ncap++ > 4000000) return;
    if (!f) f = fopen(slifter_debug_path("sop_dump.bin").c_str(), "wb");
    if (f) {
        int32_t rec[6] = {tid, m, n, k_local, a_base, b_base};
        fwrite(rec, 4, 6, f);
        fwrite(&av, 4, 1, f);
        fwrite(&bv, 4, 1, f);
        // Flush periodically, not every record (per-record fflush is the dominant
        // cost + a likely source of the heavy-dump instability).
        if ((ncap & 0x3fff) == 0) fflush(f);
    }
}

// Portable f32 -> fp16 raw-bits (round-to-nearest-even), for KID decode only.
static inline uint16_t slifter_kid_f32_to_f16bits(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);            // overflow -> inf
    if (exp <= 0) {                                                // subnormal/zero
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint16_t h = (uint16_t)(mant >> shift);
        if ((mant >> (shift - 1)) & 1) h += 1;                     // round
        return (uint16_t)(sign | h);
    }
    uint16_t h = (uint16_t)((exp << 10) | (mant >> 13));
    if (mant & 0x1000u) h += 1;                                    // round-to-nearest
    return (uint16_t)(sign | h);
}

// Diagnostic: dump the hd96 dQ K-read (dq_kread) — (n_val=hd, kc=key-in-MMA, b_base,
// b_off, bv). With SLIFTER_KID_INJECT the K smem holds raw uint16 ids ((key<<7)|hd),
// so bv re-encoded to fp16 bits recovers the (key,hd) the read actually landed on.
// NOISE-FREE decode of read_hd vs intended n_val. Gated SLIFTER_KID_INJECT.
extern "C" void slifter_dqk_dump(int32_t n_val, int32_t kc, int32_t b_base,
                                 int32_t b_kg0, int32_t b_off, float bv) {
    if (!getenv("SLIFTER_KID_INJECT")) return;
    const int tid = g_threadCtx.tid;
    if (tid / 128 != 1 || g_threadCtx.cta != 0) return;  // one consumer warpgroup, cta0
    static FILE* f = nullptr;
    static std::mutex mtx;
    static long ncap = 0;
    std::lock_guard<std::mutex> lk(mtx);
    if (ncap++ > 4000000) return;
    if (!f) f = fopen(slifter_debug_path("dqkbase_dump.bin").c_str(), "wb");
    if (f) {
        // record: [n_val(H), kc(KC), b_base(BB), b_kg0(BKG), b_off(BO), id(KID)]
        // recover the injected raw id bits: bv == fp16_decode(idbits); re-encode bv->fp16
        // (portable f32->fp16-bits, round-to-nearest, no F16C/_Float16 dependency).
        uint16_t kid = slifter_kid_f32_to_f16bits(bv);
        int32_t rec[6] = {n_val, kc, b_base, b_kg0, b_off, (int32_t)kid};
        fwrite(rec, 4, 6, f);
        if ((ncap & 0x3fff) == 0) fflush(f);
    }
}

// Diagnostic: dump the PV (RS-form) WGMMA decode accumulator per (m,n), aggregated
// by hdim block, so we can see whether the decode produces nonzero acc for the hdim
// blocks that end up 0 in O. Gated by SLIFTER_PVACC_DUMP.
extern "C" void slifter_pvacc_dump(int32_t m, int32_t n, float v) {
    if (!getenv("SLIFTER_PVACC_DUMP")) return;
    static std::atomic<long> nz[4]{}, zr[4]{}, total{0};
    if (g_threadCtx.cta == 0 && g_threadCtx.tid == 128) {
        int blk = (n >> 5) & 3;
        if (v != 0.0f) nz[blk].fetch_add(1); else zr[blk].fetch_add(1);
        long t = total.fetch_add(1) + 1;
        // print frequently (small total per thread) + dump raw records early.
        if (t <= 130)
            std::fprintf(stderr, "[pvacc] #%ld m=%d n=%d blk=%d v=%g\n", t, m, n, blk, (double)v);
        if (t % 200 == 0)
            std::fprintf(stderr, "[pvacc-sum @%ld] blk0 nz=%ld z=%ld | blk1 nz=%ld z=%ld | blk2 nz=%ld z=%ld | blk3 nz=%ld z=%ld\n",
                t, nz[0].load(),zr[0].load(), nz[1].load(),zr[1].load(),
                nz[2].load(),zr[2].load(), nz[3].load(),zr[3].load());
    }
}

// Diagnostic: dump the bwd softmax EX2 input (S*scale_log2 - lse) and output (P)
// for cta0/wg1 (capped), to pin why the recomputed P reads 1.0 in the dV GEMM.
// Gated by SLIFTER_EX2_DUMP.
extern "C" void slifter_ex2_dump(float in, float out) {
    if (!getenv("SLIFTER_EX2_DUMP")) return;
    if (g_threadCtx.cta != 0 || (g_threadCtx.tid / 128) != 1) return;
    static std::atomic<int> n{0};
    int c = n.fetch_add(1);
    if (c < 64)
        std::fprintf(stderr, "[ex2] tid=%d #%d  in(S*sc-lse)=%.5f  out(P)=%.6f\n",
                     g_threadCtx.tid, c, in, out);
}

// Diagnostic: per-lane O-store (ST.E) address, to decode the packgqa O-store column
// mapping. Gated by SLIFTER_OSTORE_DBG; width=4 is the ST.E.128 (O) store, 1 = LSE.
extern "C" void slifter_ostore_dbg(int64_t addr, int32_t width) {
    if (!getenv("SLIFTER_OSTORE_DBG")) return;
    static std::atomic<int> n{0};
    if (g_threadCtx.cta == 0 && n.fetch_add(1) < 8000)
        std::fprintf(stderr, "[ostore] cta=%d tid=%d lane=%d w=%d addr=%lld\n",
                     g_threadCtx.cta, g_threadCtx.tid, g_threadCtx.lane, width,
                     (long long)addr);
}

// Diagnostic: per-thread LDGSTS gather (global byte address -> shared slot), to
// decode the packgqa Q-gather M-row->query mapping. Gated by SLIFTER_LDGSTS_DBG.
// SLIFTER_LDGSTS_QBASE (decimal pointer) lets us print (addr-qbase) as a query row.
extern "C" void slifter_ldgsts_dbg(int64_t global_addr, int32_t shared_off) {
    if (!getenv("SLIFTER_LDGSTS_DBG")) return;
    static std::atomic<int> n{0};
    if (g_threadCtx.cta != 0) return;
    if (n.fetch_add(1) >= 4000) return;
    long long qbase = 0;
    if (const char* e = getenv("SLIFTER_LDGSTS_QBASE")) qbase = atoll(e);
    long long rel = (long long)global_addr - qbase;
    // [B,SQ,H,D] fp16 row-major: row stride (over sq, for fixed b/h) = H*D*2 = 1024 B
    long long qrow = (qbase && rel >= 0) ? (rel / 1024) : -1;
    std::fprintf(stderr, "[ldgsts] cta=%d tid=%d lane=%d wg=%d soff=%d gaddr=%lld rel=%lld qrow=%lld\n",
                 g_threadCtx.cta, g_threadCtx.tid, g_threadCtx.lane, g_threadCtx.tid/128,
                 shared_off, (long long)global_addr, rel, qrow);
}

extern "C" void named_barrier_arrive(uint32_t id, uint32_t count) {
    if (!warp_fiber_active() || !g_namedBarCtx) return;          // per-thread: NOP
    NamedBarCtx& ctx = g_namedBarCtx[(size_t)g_threadCtx.cta * SLIFTER_MAX_NAMED_BAR
                                     + (id & (SLIFTER_MAX_NAMED_BAR - 1))];
    int a = ++ctx.arrived;
    if (nbar_dbg() && a == 1)
        std::fprintf(stderr, "[nbar] ARV enter id=%u count=%u tid=%d\n", id, count, g_threadCtx.tid);
    if (a >= (int)count) {
        if (nbar_dbg()) std::fprintf(stderr, "[nbar] ARV COMPLETE id=%u count=%u tid=%d\n", id, count, g_threadCtx.tid);
        ctx.arrived = 0; ++ctx.generation;  // release waiters
    }
}

// -----------------------------------------------------------------------------
// vote_any
// Matches IR:
//   declare i8 @"vote_any"(i32 %mask, i8 %pred, i32 %site)
// -----------------------------------------------------------------------------

extern "C" bool vote_any(uint32_t mask, bool pred, uint32_t site)
{
    (void)mask;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return pred;
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, 0, pred, vals, preds);
    uint32_t grp_ballot = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && preds[M]) grp_ballot |= (1u << M);
    return grp_ballot != 0;
}

// -----------------------------------------------------------------------------
// vote_all
// Matches IR:
//   declare i8 @"vote_all"(i32 %mask, i8 %pred, i32 %site)
// -----------------------------------------------------------------------------

extern "C" bool vote_all(uint32_t mask, bool pred, uint32_t site)
{
    (void)mask;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return pred;
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, 0, pred, vals, preds);
    uint32_t grp_ballot = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && preds[M]) grp_ballot |= (1u << M);
    return grp_ballot == members;
}

// -----------------------------------------------------------------------------
// vote_ballot
// Matches IR:
//   declare i32 @"vote_ballot"(i32 %mask, i8 %pred, i32 %site)
// Returns a bitmask of lanes where pred is true.
// -----------------------------------------------------------------------------

extern "C" uint32_t vote_ballot(uint32_t mask, bool pred, uint32_t site)
{
    (void)mask;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1)
        return pred ? (1u << g_threadCtx.lane) : 0u;

    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, 0, pred, vals, preds);
    uint32_t ballot = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && preds[M]) ballot |= (1u << M);
    return ballot;
}

// -----------------------------------------------------------------------------
// hmma16816_f32_warp — warp-exchange tensor-core emulation
//
// HMMA.16816.F32 computes D(16x8,f32) = A(16x16,f16) x B(16x8,f16) + C with
// fragments spread across the warp (PTX mma.m16n8k16 layout: lane = 4*g+t,
// g=lane>>2 group/row, t=lane&3 quad position covering K). Per-lane scalar
// decomposition can only see 4 of the 16 K products, so the runtime gathers
// every lane's fragments, computes the full tile once, and hands each lane
// its D slice. Uniform warps only (WMMA code has no divergence around the
// MMA), so arrival-based instance grouping is exact here.
// Layout (PTX ISA, mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32):
//   a0={A[g][2t],A[g][2t+1]}        a1={A[g+8][2t],A[g+8][2t+1]}
//   a2={A[g][2t+8],A[g][2t+9]}      a3={A[g+8][2t+8],A[g+8][2t+9]}
//   b0={B[2t][g],B[2t+1][g]}        b1={B[2t+8][g],B[2t+9][g]}
//   c/d k: c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// movm_mt88_warp — MOVM.16.MT88 (PTX movmatrix.sync.m8n8.trans.b16)
// The warp's 32 regs form an 8x8 b16 matrix: lane L holds row L/4,
// cols 2*(L&3) and 2*(L&3)+1. Returns each lane's pair of the TRANSPOSED
// matrix. Uniform warps (ptxas emits MOVM only in tensor-core tiles).
// -----------------------------------------------------------------------------

struct WarpMovmCtx {
    std::mutex m;
    std::condition_variable cv;
    int   live       = 0;
    int   arrived    = 0;
    int   generation = 0;
    uint32_t vals[WARP_SIZE];
    uint32_t results[WARP_SIZE];
};
static WarpMovmCtx* g_warpMovmCtx = nullptr;

extern "C" uint32_t movm_mt88_warp(uint32_t v)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpMovmCtx& ctx = g_warpMovmCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.vals[lane] = v;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        uint16_t M[8][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int r = ln >> 2, c = 2 * (ln & 3);
            M[r][c]     = (uint16_t)(ctx.vals[ln] & 0xffff);
            M[r][c + 1] = (uint16_t)(ctx.vals[ln] >> 16);
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int r = ln >> 2, c = 2 * (ln & 3);
            ctx.results[ln] = (uint32_t)M[c][r] | ((uint32_t)M[c + 1][r] << 16);
        }
    });
    return ctx.results[lane];
}

struct WarpHmmaCtx {
    std::mutex m;
    std::condition_variable cv;
    int   live       = 0;   // unused (uniform warps) but kept for symmetry
    int   arrived    = 0;
    int   generation = 0;
    uint32_t a[WARP_SIZE][4];
    uint32_t bfrag[WARP_SIZE][2];
    float    c[WARP_SIZE][4];
    float    d[WARP_SIZE][4];
};
static WarpHmmaCtx* g_warpHmmaCtx = nullptr;

static inline float half_bits_to_f32_(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) out = sign;
        else { int e = -1; do { e++; mant <<= 1; } while (!(mant & 0x400));
               mant &= 0x3ff; out = sign | ((uint32_t)(127-15-e) << 23) | (mant << 13); }
    } else if (exp == 31) out = sign | 0x7f800000u | (mant << 13);
    else out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

extern "C" void hmma16816_f32_warp(float* dout,
                                   uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                   uint32_t b0, uint32_t b1,
                                   float c0, float c1, float c2, float c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1; ctx.a[lane][2]=a2; ctx.a[lane][3]=a3;
    ctx.bfrag[lane][0]=b0; ctx.bfrag[lane][1]=b1;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][16], B[16][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 4; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = (r & 1) ? g + 8 : g;
                const int col = (r & 2) ? 2*t + 8 : 2*t;
                A[row][col]     = half_bits_to_f32_(lo);
                A[row][col + 1] = half_bits_to_f32_(hi);
            }
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][r] >> 16);
                const int krow = (r ? 2*t + 8 : 2*t);
                B[krow][g]     = half_bits_to_f32_(lo);
                B[krow + 1][g] = half_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 16; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

static inline float bf16_bits_to_f32_(uint16_t h) {
    // bfloat16 is the top 16 bits of an f32 — exact widen.
    uint32_t out = (uint32_t)h << 16;
    float f; std::memcpy(&f, &out, 4); return f;
}

// hmma16816_bf16_warp — HMMA.16816.F32.BF16: warp-true BF16-input MMA.
// Identical staging/finalize to hmma16816_f32_warp (same m16n8k16 A/B
// fragment layout, same f32 C/D ABI); only the A/B fragment decode
// differs (packed bf16 pairs instead of f16 pairs).
extern "C" void hmma16816_bf16_warp(float* dout,
                                    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1,
                                    float c0, float c1, float c2, float c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1; ctx.a[lane][2]=a2; ctx.a[lane][3]=a3;
    ctx.bfrag[lane][0]=b0; ctx.bfrag[lane][1]=b1;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][16], B[16][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 4; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = (r & 1) ? g + 8 : g;
                const int col = (r & 2) ? 2*t + 8 : 2*t;
                A[row][col]     = bf16_bits_to_f32_(lo);
                A[row][col + 1] = bf16_bits_to_f32_(hi);
            }
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][r] >> 16);
                const int krow = (r ? 2*t + 8 : 2*t);
                B[krow][g]     = bf16_bits_to_f32_(lo);
                B[krow + 1][g] = bf16_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 16; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

static inline uint16_t f32_to_half_bits_(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t  exp  = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00 | (exp == 143 && mant ? 0x200 : 0));
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t h = mant >> shift;
        uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)(sign | (uint32_t)(exp << 10) | (mant >> 13));
    uint32_t rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}

// Half <-> float soft-conversion libcalls. Lifted IR with scalar `half`
// arithmetic lowers to these on plain x86_64 (no f16c target feature);
// clang expects compiler-rt builtins, and this toolchain's g++/libgcc
// doesn't provide them (same situation as __truncsfbf2 above). Surfaced
// when the MMA width-true typing rebalanced which half ops remain in IR
// (cudnn_cnn so.26/33/40/61 link_fail, recount 142513).
extern "C" float __gnu_h2f_ieee(uint16_t h) { return half_bits_to_f32_(h); }
extern "C" uint16_t __gnu_f2h_ieee(float f) { return f32_to_half_bits_(f); }
// __extendhfsf2/__truncsfhf2: clang calls these with the _Float16 ABI —
// the half travels in XMM0's low 16 bits, NOT an integer register. This
// g++ (11.x) has no _Float16 on x86-64, so use a float bit-carrier: a
// float param/return also lives in XMM0, and only its low 16 bits are
// read/written as the half. The uint16_t signatures returned the bits in
// EAX while callers read stale XMM0 — every soft-half conversion
// produced garbage (f2fp_e4m3_unpack: small f16 results read back as 0).
extern "C" float __extendhfsf2(float hbits) {
    uint32_t b;
    std::memcpy(&b, &hbits, 4);
    return half_bits_to_f32_((uint16_t)b);
}
extern "C" float __truncsfhf2(float f) {
    uint32_t b = f32_to_half_bits_(f);
    float r;
    std::memcpy(&r, &b, 4);
    return r;
}

// hmma16816_f16_warp — HMMA.16816.F16: warp-true F16-accum tensor-core MMA.
// Same staging/finalize as hmma16816_f32_warp (identical m16n8k16 A/B
// layout); C arrives as 2 packed v2f16 regs (4 halves) and D returns the
// same way. Accumulation in f32, single round to f16 at the end (PTX: mma
// f16-out accumulates in at least single precision, then converts).
extern "C" void hmma16816_f16_warp(uint32_t* dout,
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t b0, uint32_t b1,
        uint32_t c0, uint32_t c1)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1; ctx.a[lane][2]=a2; ctx.a[lane][3]=a3;
    ctx.bfrag[lane][0]=b0; ctx.bfrag[lane][1]=b1;
    ctx.c[lane][0]=half_bits_to_f32_((uint16_t)(c0 & 0xffff));
    ctx.c[lane][1]=half_bits_to_f32_((uint16_t)(c0 >> 16));
    ctx.c[lane][2]=half_bits_to_f32_((uint16_t)(c1 & 0xffff));
    ctx.c[lane][3]=half_bits_to_f32_((uint16_t)(c1 >> 16));
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][16], B[16][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 4; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = (r & 1) ? g + 8 : g;
                const int col = (r & 2) ? 2*t + 8 : 2*t;
                A[row][col]     = half_bits_to_f32_(lo);
                A[row][col + 1] = half_bits_to_f32_(hi);
            }
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][r] >> 16);
                const int krow = (r ? 2*t + 8 : 2*t);
                B[krow][g]     = half_bits_to_f32_(lo);
                B[krow + 1][g] = half_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 16; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    const uint16_t d0 = f32_to_half_bits_(ctx.d[lane][0]);
    const uint16_t d1 = f32_to_half_bits_(ctx.d[lane][1]);
    const uint16_t d2 = f32_to_half_bits_(ctx.d[lane][2]);
    const uint16_t d3 = f32_to_half_bits_(ctx.d[lane][3]);
    dout[0] = (uint32_t)d0 | ((uint32_t)d1 << 16);
    dout[1] = (uint32_t)d2 | ((uint32_t)d3 << 16);
}

// -----------------------------------------------------------------------------
// hmma1688_*_warp — HMMA.1688 family (m16n8k8): warp-true tensor-core MMA.
// Same one-barrier staging design as hmma16816_f32_warp, reusing the same
// per-warp WarpHmmaCtx (the 16816-sized fragment arrays are big enough; the
// 1688 forms use a smaller footprint). Fragment layouts derived from cute
// MMA_Traits<SM80_16x8x8_*> and confirmed against the fixture SASS
// (test/sm90_synth/hmma1688{tf32,bf16,f32,f16}); lane = 4g+t:
//   f16/bf16 in:  a0={A[g][2t],A[g][2t+1]}, a1={A[g+8][2t],A[g+8][2t+1]}
//                 b0={B[2t][g],B[2t+1][g]}          (= the k<8 half of 16816)
//   tf32 in:      a0=A[g][t], a1=A[g+8][t], a2=A[g][t+4], a3=A[g+8][t+4]
//                 b0=B[t][g], b1=B[t+4][g]          (one tf32 per 32-bit reg)
//   c/d (all):    c0=C[g][2t] c1=C[g][2t+1] c2=C[g+8][2t] c3=C[g+8][2t+1]
// -----------------------------------------------------------------------------

static inline float tf32_bits_to_f32_(uint32_t u) {
    // tf32 = 1s + 8e + 10m in the upper bits of a 32-bit reg; the tensor
    // core ignores the low 13 mantissa bits (real kernels feed cvt-converted
    // values, so the mask is an identity there).
    u &= 0xFFFFE000u;
    float f; std::memcpy(&f, &u, 4); return f;
}

// hmma1688_tf32_warp — HMMA.1688.F32.TF32: the standard Ampere f32 tensor
// path (cutlass ex14/27/28/29/33/59, cuBLAS tf32 sgemm kernels).
extern "C" void hmma1688_tf32_warp(float* dout,
        float a0, float a1, float a2, float a3,
        float b0, float b1,
        float c0, float c1, float c2, float c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    const float av[4] = {a0, a1, a2, a3};
    const float bv[2] = {b0, b1};
    std::memcpy(ctx.a[lane], av, sizeof(av));
    std::memcpy(ctx.bfrag[lane], bv, sizeof(bv));
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][8], B[8][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 4; ++r) {
                const int row = (r & 1) ? g + 8 : g;
                const int col = (r & 2) ? t + 4 : t;
                A[row][col] = tf32_bits_to_f32_(ctx.a[ln][r]);
            }
            B[t][g]     = tf32_bits_to_f32_(ctx.bfrag[ln][0]);
            B[t + 4][g] = tf32_bits_to_f32_(ctx.bfrag[ln][1]);
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 8; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

// hmma1688_f32_warp — HMMA.1688.F32: f16-input/f32-accum m16n8k8
// (cutlass ex12 gemm_bias_relu, ex41 attention forward).
extern "C" void hmma1688_f32_warp(float* dout,
        uint32_t a0, uint32_t a1, uint32_t b0,
        float c0, float c1, float c2, float c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1;
    ctx.bfrag[lane][0]=b0;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][8], B[8][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = r ? g + 8 : g;
                A[row][2*t]     = half_bits_to_f32_(lo);
                A[row][2*t + 1] = half_bits_to_f32_(hi);
            }
            {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][0] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][0] >> 16);
                B[2*t][g]     = half_bits_to_f32_(lo);
                B[2*t + 1][g] = half_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 8; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

// hmma1688_bf16_warp — HMMA.1688.F32.BF16: bf16-input/f32-accum m16n8k8
// (cutlass ex41 backward). Identical staging/finalize to hmma1688_f32_warp;
// only the A/B fragment decode differs (packed bf16 pairs).
extern "C" void hmma1688_bf16_warp(float* dout,
        uint32_t a0, uint32_t a1, uint32_t b0,
        float c0, float c1, float c2, float c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1;
    ctx.bfrag[lane][0]=b0;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][8], B[8][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = r ? g + 8 : g;
                A[row][2*t]     = bf16_bits_to_f32_(lo);
                A[row][2*t + 1] = bf16_bits_to_f32_(hi);
            }
            {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][0] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][0] >> 16);
                B[2*t][g]     = bf16_bits_to_f32_(lo);
                B[2*t + 1][g] = bf16_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 8; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

// hmma1688_f16_warp — HMMA.1688.F16: f16-accum m16n8k8 (cutlass ex13 fused
// gemm/conv chains). Same A/B staging as hmma1688_f32_warp; C arrives as 2
// packed v2f16 regs (4 halves) and D returns the same way. Accumulation in
// f32, single round to f16 at the end (PTX: mma f16-out accumulates in at
// least single precision, then converts) — same policy as
// hmma16816_f16_warp.
extern "C" void hmma1688_f16_warp(uint32_t* dout,
        uint32_t a0, uint32_t a1, uint32_t b0,
        uint32_t c0, uint32_t c1)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpHmmaCtx& ctx = g_warpHmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1;
    ctx.bfrag[lane][0]=b0;
    ctx.c[lane][0]=half_bits_to_f32_((uint16_t)(c0 & 0xffff));
    ctx.c[lane][1]=half_bits_to_f32_((uint16_t)(c0 >> 16));
    ctx.c[lane][2]=half_bits_to_f32_((uint16_t)(c1 & 0xffff));
    ctx.c[lane][3]=half_bits_to_f32_((uint16_t)(c1 >> 16));
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        float A[16][8], B[8][8], C[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int r = 0; r < 2; ++r) {
                const uint16_t lo = (uint16_t)(ctx.a[ln][r] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.a[ln][r] >> 16);
                const int row = r ? g + 8 : g;
                A[row][2*t]     = half_bits_to_f32_(lo);
                A[row][2*t + 1] = half_bits_to_f32_(hi);
            }
            {
                const uint16_t lo = (uint16_t)(ctx.bfrag[ln][0] & 0xffff);
                const uint16_t hi = (uint16_t)(ctx.bfrag[ln][0] >> 16);
                B[2*t][g]     = half_bits_to_f32_(lo);
                B[2*t + 1][g] = half_bits_to_f32_(hi);
            }
            C[g][2*t]       = ctx.c[ln][0];
            C[g][2*t + 1]   = ctx.c[ln][1];
            C[g + 8][2*t]   = ctx.c[ln][2];
            C[g + 8][2*t+1] = ctx.c[ln][3];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            const int rows[2] = {g, g + 8};
            const int cols[2] = {2*t, 2*t + 1};
            for (int ri = 0; ri < 2; ++ri)
                for (int ci = 0; ci < 2; ++ci) {
                    float acc = C[rows[ri]][cols[ci]];
                    for (int k = 0; k < 8; ++k)
                        acc += A[rows[ri]][k] * B[k][cols[ci]];
                    ctx.d[ln][2*ri + ci] = acc;
                }
        }
    });

    const uint16_t d0 = f32_to_half_bits_(ctx.d[lane][0]);
    const uint16_t d1 = f32_to_half_bits_(ctx.d[lane][1]);
    const uint16_t d2 = f32_to_half_bits_(ctx.d[lane][2]);
    const uint16_t d3 = f32_to_half_bits_(ctx.d[lane][3]);
    dout[0] = (uint32_t)d0 | ((uint32_t)d1 << 16);
    dout[1] = (uint32_t)d2 | ((uint32_t)d3 << 16);
}

// -----------------------------------------------------------------------------
// ldsm_m88_warp — LDSM.16.M88[.N] / LDSM.16.MT88[.N] (PTX ldmatrix.m8n8.bN)
//
// ldmatrix is warp-collective: each lane supplies the address of one 16-byte
// matrix ROW (lanes 8m..8m+7 address the 8 rows of matrix m), but the loaded
// fragments are redistributed across the warp in the standard mma b16 layout:
//   reg m of lane L = 4-byte slice (L&3) of the row addressed by lane 8m+(L>>2)
// .trans (MT88) hands each lane the fragment of the TRANSPOSED matrix:
//   reg m of lane L = halfword (L>>2) of row 2*(L&3)   (lo 16 bits)
//                   | halfword (L>>2) of row 2*(L&3)+1 (hi 16 bits)
// A per-lane contiguous load from the lane's own address is a fragment
// permutation of the warp (verified against the ldsm_x4 GPU capture: only
// 8/512 elements coincide) — tensor-core kernels then consume permuted A/B.
//
// The lifted IR loads the full 16-byte row at each lane's own address and
// parks it here; the last arriver redistributes. For .x1/.x2 the unused
// lanes' rows are parked but ignored, mirroring "addresses of remaining
// threads are ignored" (PTX) — their own-row loads still happen in the IR.
// -----------------------------------------------------------------------------

struct WarpLdsmCtx {
    std::mutex m;
    std::condition_variable cv;
    int   live       = 0;
    int   arrived    = 0;
    int   generation = 0;
    uint32_t row[WARP_SIZE][4];
    uint32_t out[WARP_SIZE][4];
};
static WarpLdsmCtx* g_warpLdsmCtx = nullptr;

extern "C" void ldsm_m88_warp(uint32_t* dout,
        uint32_t r0, uint32_t r1, uint32_t r2, uint32_t r3,
        int count, int trans)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpLdsmCtx& ctx = g_warpLdsmCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.row[lane][0]=r0; ctx.row[lane][1]=r1;
    ctx.row[lane][2]=r2; ctx.row[lane][3]=r3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int m = 0; m < count; ++m) {
                if (!trans) {
                    ctx.out[ln][m] = ctx.row[8*m + g][t];
                } else {
                    const uint32_t w0 = ctx.row[8*m + 2*t][g >> 1];
                    const uint32_t w1 = ctx.row[8*m + 2*t + 1][g >> 1];
                    const uint16_t lo = (uint16_t)((g & 1) ? (w0 >> 16) : (w0 & 0xffff));
                    const uint16_t hi = (uint16_t)((g & 1) ? (w1 >> 16) : (w1 & 0xffff));
                    ctx.out[ln][m] = (uint32_t)lo | ((uint32_t)hi << 16);
                }
            }
        }
    });

    for (int k = 0; k < count; ++k) dout[k] = ctx.out[lane][k];
}

// -----------------------------------------------------------------------------
// stsm_m88_warp — STSM.16.M88[.N] / STSM.16.MT88[.N] (PTX stmatrix.m8n8.bN)
//
// stmatrix is the warp-collective INVERSE of ldmatrix: lanes 8m..8m+7 supply
// the addresses of the 8 rows of matrix m, and the 16-byte row written at
// lane (8m+r)'s address gathers fragments from ACROSS the warp:
//   word w of row r (matrix m) = src reg m of lane 4r+w
// .trans (MT88) stores the TRANSPOSE of the fragment matrix:
//   word w of row r = sel(r&1, src_m[8w+(r>>1)]) | sel(r&1, src_m[8w+4+(r>>1)]) << 16
//   (sel(0)=lo16, sel(1)=hi16) — the exact inverse of ldsm_m88_warp's MT88
// gather, so ldsm(stsm(frags)) round-trips identity in both forms.
//
// Each lane parks its N source regs and its OWN target-row pointer; the last
// arriver writes the redistributed rows through the parked pointers. Only
// lanes 0..8*count-1 have a target — the addresses of the remaining lanes
// are IGNORED per the PTX ISA, and (unlike ldmatrix's harmless extra loads)
// dereferencing them would corrupt memory, so all stores happen here and
// the lifted IR performs none itself.
// -----------------------------------------------------------------------------

struct WarpStsmCtx {
    std::mutex m;
    std::condition_variable cv;
    int   live       = 0;
    int   arrived    = 0;
    int   generation = 0;
    uint32_t  src[WARP_SIZE][4];
    uint32_t* dst[WARP_SIZE];
};
static WarpStsmCtx* g_warpStsmCtx = nullptr;

// fp8 (1 byte/elem) flag: when set, stsm_m88_warp gathers at BYTE granularity so
// the FA3 in-kernel V-transpose chain `LDSM.16.MT88(b16) + PRMT(0x6420/0x7531
// even/odd byte de-interleave) + STSM.16.M88` nets a CORRECT 8-bit transpose.
// The b16 MT88 keeps each (hdim 2j, hdim 2j+1) pair inside one 16-bit unit; the
// PRMT splits even/odd bytes into separate src regs; this gather then routes
// those de-interleaved bytes to the right hdim rows. Without it the net is a
// rotate-left-1 over the low 4 hdim bits within each 16-hdim transpose unit (the
// FA3 fp8 "hdim blocks 0,2 ok / 1,3 zero" signature). Derived and verified
// over all 512 cases.
//   For dst hdim row dh=8m+r, word w (4 seqlen bytes ds=4w..4w+3):
//     reg  = 2*(dh>>4) + (dh&1)                 (which PRMT'd src reg)
//     laneA= 4*((dh&15)>>1) + 2*(w&1), laneB = laneA+1
//     bb   = 2*(w>>1)                            (byte hi-half within word)
//     word = src[laneA][reg].byte[bb]   | src[laneA][reg].byte[bb+1]<<8
//          | src[laneB][reg].byte[bb]<<16| src[laneB][reg].byte[bb+1]<<24
extern "C" void stsm_m88_warp(uint32_t* dst,
        uint32_t s0, uint32_t s1, uint32_t s2, uint32_t s3,
        int count, int trans)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpStsmCtx& ctx = g_warpStsmCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.src[lane][0]=s0; ctx.src[lane][1]=s1;
    ctx.src[lane][2]=s2; ctx.src[lane][3]=s3;
    ctx.dst[lane] = dst;
    ctx.arrived++;
    if (getenv("SLIFTER_STSM_DBG")) {
        static std::atomic<int> n{0};
        if (g_threadCtx.cta == 0 && n.fetch_add(1) < 6000) {
            long doff = (long)((const char*)dst - (const char*)shared_mem);
            std::fprintf(stderr, "[stsm] tid=%d lane=%d count=%d dst_off=%ld dst=%p src=%u,%u,%u,%u trans=%d\n",
                         g_threadCtx.tid, lane, count, doff, (void*)dst, s0, s1, s2, s3, trans);
        }
    }

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        // trans==2 selects the fp8 (1-byte) transpose; ==1 the b16 transpose;
        // ==0 the non-transposed gather (fp16/bf16 are byte-identical: they
        // never pass trans==2).
        if (trans == 2) {
            for (int m = 0; m < count; ++m) {
                for (int r = 0; r < 8; ++r) {
                    uint32_t* out = ctx.dst[8*m + r];
                    const int dh = 8*m + r;
                    const int reg = 2*(dh >> 4) + (dh & 1);
                    for (int w = 0; w < 4; ++w) {
                        const int laneA = 4*((dh & 15) >> 1) + 2*(w & 1);
                        const int laneB = laneA + 1;
                        const int bb = 2*(w >> 1);
                        const uint8_t* a = (const uint8_t*)&ctx.src[laneA][reg];
                        const uint8_t* b = (const uint8_t*)&ctx.src[laneB][reg];
                        out[w] = (uint32_t)a[bb] | ((uint32_t)a[bb+1] << 8)
                               | ((uint32_t)b[bb] << 16) | ((uint32_t)b[bb+1] << 24);
                    }
                }
            }
            return;
        }
        for (int m = 0; m < count; ++m) {
            for (int r = 0; r < 8; ++r) {
                uint32_t* out = ctx.dst[8*m + r];
                for (int w = 0; w < 4; ++w) {
                    if (!trans) {
                        out[w] = ctx.src[4*r + w][m];
                    } else {
                        const uint32_t v0 = ctx.src[8*w + (r >> 1)][m];
                        const uint32_t v1 = ctx.src[8*w + 4 + (r >> 1)][m];
                        const uint16_t lo = (uint16_t)((r & 1) ? (v0 >> 16) : (v0 & 0xffff));
                        const uint16_t hi = (uint16_t)((r & 1) ? (v1 >> 16) : (v1 & 0xffff));
                        out[w] = (uint32_t)lo | ((uint32_t)hi << 16);
                    }
                }
            }
        }
        // DIAG (SLIFTER_STSM_SMEM): after the store, record each written 16B row's
        // smem byte offset + 4 words. For the dS STSM.MT88 (dest in the 0x2e800
        // region) this lets the analyzer reconstruct the physical dS smem layout and
        // compare to CPU dS via SmemLayoutPdS — value-checks the transpose store on
        // the COMMITTED binary (wrapper-only, no re-lift). cta0 only.
        if (getenv("SLIFTER_STSM_SMEM") && g_threadCtx.cta == 0 && trans == 1) {
            static FILE* sf = nullptr; static std::mutex smx; std::lock_guard<std::mutex> lk(smx);
            if (!sf) sf = fopen(slifter_debug_path("stsm_smem.bin").c_str(), "wb");
            if (sf) for (int m = 0; m < count; ++m) for (int r = 0; r < 8; ++r) {
                uint32_t* out = ctx.dst[8*m + r];
                long doff = (long)((const char*)out - (const char*)shared_mem);
                int32_t hdr[2] = {(int32_t)doff, g_threadCtx.tid};
                fwrite(hdr,4,2,sf); fwrite(out,4,4,sf);
            }
        }
        // DIAG (SLIFTER_STSM_SRC): record the per-lane source registers (warp,lane,reg -> u32)
        // for the FIRST few cta0 trans==1 STSM calls so the dS REGISTER fragment can be value-
        // checked against CPU dS at the SdP CLayout (key,query) — isolates store-side vs the
        // upstream dS value/layout. Runtime-env-gated.
        if (getenv("SLIFTER_STSM_SRC") && g_threadCtx.cta == 0 && trans == 1) {
            static FILE* rf = nullptr; static std::mutex rmx; static int rcalls = 0;
            std::lock_guard<std::mutex> lk(rmx);
            if (!rf) rf = fopen(slifter_debug_path("stsm_src.bin").c_str(), "wb");
            if (rf && rcalls < 10) {
                for (int lane = 0; lane < WARP_SIZE; ++lane)
                    for (int reg = 0; reg < count; ++reg) {
                        int32_t rec[4] = {rcalls, lane, reg, (int32_t)ctx.src[lane][reg]};
                        fwrite(rec,4,4,rf);
                    }
                rcalls++;
            }
        }
        // DIAG (SLIFTER_STSM_ADDR): record the per-lane row address (lane,m,r -> doff) for the
        // FIRST few cta0 trans==1 STSM calls so the addresses can be compared to cute's expected
        // SmemLayoutPdSt store positions. Lift-time unaffected (runtime-env-gated).
        if (getenv("SLIFTER_STSM_ADDR") && g_threadCtx.cta == 0 && trans == 1) {
            static FILE* af = nullptr; static std::mutex amx; static int acalls = 0;
            std::lock_guard<std::mutex> lk(amx);
            if (!af) af = fopen(slifter_debug_path("stsm_addr.bin").c_str(), "wb");
            if (af && acalls < 5) {
                for (int lane = 0; lane < WARP_SIZE; ++lane) {
                    long doff = (long)((const char*)ctx.dst[lane] - (const char*)shared_mem);
                    int32_t rec[3] = {acalls, lane, (int32_t)doff};
                    fwrite(rec,4,3,af);
                }
                acalls++;
            }
        }
    });
}

// -----------------------------------------------------------------------------
// dmma_16816_f64 — DMMA.16x8x16: faithful warp-true F64 tensor-core MMA.
// Matches IR:
//   declare void @"dmma_16816_f64"(double* %d, double x16)   ; 8 A, 4 B, 4 C
//
// Same one-barrier staging design as hmma16816_f32_warp: every lane parks
// its fragments in the per-warp context, the last arriver reconstructs the
// full 16x16 A / 16x8 B and computes all lanes' D, everyone reads back its
// own 4 values. Fragment->matrix mapping is the CUTLASS cute
// MMA_Traits<SM90_16x8x16_F64F64F64F64_TN> layout (decode convention
// cross-checked against the canonical m16n8 C-accumulator mapping):
//   lane = 4*g + t   (g = groupID, t = threadID-in-group)
//   a[v0 + 2*v1] = A[g + 8*v0][t + 4*v1]   (v0 in {0,1}, v1 in {0..3})
//   b[v]         = B[t + 4*v][g]           (B is K x N)
//   c/d[v0+2*v1] = C[g + 8*v1][2*t + v0]
// std::fma per k-step mirrors the HW DFMA chain.
// -----------------------------------------------------------------------------

struct WarpDmmaCtx {
    std::mutex m;
    std::condition_variable cv;
    int    live       = 0;
    int    arrived    = 0;
    int    generation = 0;
    double a[WARP_SIZE][8];
    double b[WARP_SIZE][4];
    double c[WARP_SIZE][4];
    double d[WARP_SIZE][4];
};
static WarpDmmaCtx* g_warpDmmaCtx = nullptr;

extern "C" void dmma_16816_f64(double* dout,
        double a0, double a1, double a2, double a3,
        double a4, double a5, double a6, double a7,
        double b0, double b1, double b2, double b3,
        double c0, double c1, double c2, double c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpDmmaCtx& ctx = g_warpDmmaCtx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0]=a0; ctx.a[lane][1]=a1; ctx.a[lane][2]=a2; ctx.a[lane][3]=a3;
    ctx.a[lane][4]=a4; ctx.a[lane][5]=a5; ctx.a[lane][6]=a6; ctx.a[lane][7]=a7;
    ctx.b[lane][0]=b0; ctx.b[lane][1]=b1; ctx.b[lane][2]=b2; ctx.b[lane][3]=b3;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1; ctx.c[lane][2]=c2; ctx.c[lane][3]=c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        double A[16][16], B[16][8];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int v = 0; v < 8; ++v)
                A[g + 8 * (v & 1)][t + 4 * (v >> 1)] = ctx.a[ln][v];
            for (int v = 0; v < 4; ++v)
                B[t + 4 * v][g] = ctx.b[ln][v];
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int v = 0; v < 4; ++v) {
                const int m = g + 8 * (v >> 1), n = 2 * t + (v & 1);
                double acc = ctx.c[ln][v];
                for (int k = 0; k < 16; ++k)
                    acc = std::fma(A[m][k], B[k][n], acc);
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
    dout[2]=ctx.d[lane][2]; dout[3]=ctx.d[lane][3];
}

// -----------------------------------------------------------------------------
// dmma_884_f64 — DMMA.8x8x4 (m8n8k4): faithful warp-true F64 tensor-core MMA.
// Matches IR:
//   declare void @"dmma_884_f64"(double* %d, double a0, double b0, double c0, double c1)
//
// The m8n8k4 f64 atom is a warp collective: the 8x8 D tile (2 vals/lane), the
// 8x4 A and 8x4 B fragments (1 val/lane each) are distributed across all 32
// lanes. The earlier per-lane scalar-FMA stub could not reconstruct it. Same
// one-barrier rendezvous as dmma_16816_f64: each lane parks its fragments, the
// last arriver reconstructs the full A/B and computes all lanes' D, everyone
// reads back its own 2 values. Fragment->matrix mapping is the cute
// MMA_Traits<SM80_8x8x4_F64F64F64F64_TN> layout (ALayout/BLayout = SM80_8x4,
// CLayout = SM80_8x8_Row), decoded with cute's COLUMN-MAJOR codomain and
// verified against the full multi-tile GEMM (a single 8x8 atom cannot
// distinguish the C/D m<->n convention — only multi-tile exposes it):
//   lane = 4*g + t   (g = lane/4 in 0..7, t = lane%4 in 0..3)
//   a0       = A[m=g][k=t]                (A is M8 x K4, row-major)
//   b0       = B[n=g][k=t]                (B is N8 x K4, i.e. column-major K x N)
//   c/d[v]   = C[m=g][n=2*t + v]          (v in {0,1}; C/D are M8 x N8, row-major)
//   D[m][n]  = C[m][n] + sum_{k<4} A[m][k]*B[n][k].
// std::fma per k-step mirrors the HW DFMA chain.
// -----------------------------------------------------------------------------

struct WarpDmma884Ctx {
    std::mutex m;
    std::condition_variable cv;
    int    live       = 0;
    int    arrived    = 0;
    int    generation = 0;
    double a[WARP_SIZE];      // 1 A frag / lane
    double b[WARP_SIZE];      // 1 B frag / lane
    double c[WARP_SIZE][2];   // 2 C frags / lane
    double d[WARP_SIZE][2];   // 2 D frags / lane
};
static WarpDmma884Ctx* g_warpDmma884Ctx = nullptr;

extern "C" void dmma_884_f64(double* dout,
        double a0, double b0, double c0, double c1)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpDmma884Ctx& ctx = g_warpDmma884Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane]=a0; ctx.b[lane]=b0;
    ctx.c[lane][0]=c0; ctx.c[lane][1]=c1;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        double A[8][4], B[8][4];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            A[g][t] = ctx.a[ln];   // A is M8 x K4
            B[g][t] = ctx.b[ln];   // B is N8 x K4 (col-major K x N)
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, t = ln & 3;
            for (int v = 0; v < 2; ++v) {
                const int m = g, n = 2 * t + v;
                double acc = ctx.c[ln][v];
                for (int k = 0; k < 4; ++k)
                    acc = std::fma(A[m][k], B[n][k], acc);
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0]=ctx.d[lane][0]; dout[1]=ctx.d[lane][1];
}

// -----------------------------------------------------------------------------
// imma_8816_s8_warp — IMMA.8816 (m8n8k16, int8->int32): warp-true integer MMA.
// Matches IR:
//   declare void @"imma_8816_s8_warp"(int* %d, int aPacked, int bPacked, int c0, int c1)
//
// The m8n8k16 s8 atom is a warp collective: the 8x8 D tile (2 int32 vals/lane),
// the 8x16 A and 8x16 B int8 fragments (1 reg = 4 packed s8 per lane each) are
// distributed across all 32 lanes. The earlier per-lane C->D identity stub
// dropped the A*B product entirely (08_turing & friends computed D = C). Same
// one-barrier rendezvous as dmma_884_f64: each lane parks its packed A/B reg and
// 2 C values, the last arriver unpacks + reconstructs the full int8 A[8][16] /
// B[8][16] and computes all lanes' int32 D, everyone reads back its own 2 values.
// Fragment->matrix mapping is the cute MMA_Traits<SM75_8x8x16_S32S8S8S32_TN>
// layout (ALayout/BLayout = Shape<Shape<_4,_8>,_4> Stride<Stride<_32,_1>,_8>,
// CLayout = Shape<Shape<_4,_8>,_2> Stride<Stride<_16,_1>,_8>), decoded with
// cute's COLUMN-MAJOR (M-major) codomain — the SAME convention as DMMA.8x8x4,
// verified against a multi-tile GEMM (a single 8x8 atom cannot distinguish the
// C/D m<->n convention; only multi-tile exposes it):
//   lane = 4*g + tg   (g = lane/4 in 0..7, tg = lane%4 in 0..3)
//   a[v] = A[m=g][k=tg*4 + v]   v in 0..3   (A is M8 x K16, row-major)
//   b[v] = B[n=g][k=tg*4 + v]   v in 0..3   (B is N8 x K16, i.e. col-major K x N)
//   c/d[v] = C[m=g][n=tg*2 + v] v in 0..1   (C/D are M8 x N8, int32)
//   D[m][n] = C[m][n] + sum_{k<16} A[m][k]*B[n][k]   (signed int8 products, s32 accum)
// int8 byte order in a 32-bit reg: byte v at bits [8v .. 8v+7] (little-endian,
// matches PTX __dp4a packing). Accumulation is exact int32 (no rounding); the
// HW .satfinite cannot trip for these GEMM ranges (and CUTLASS picks ranges so
// it never saturates), so a plain int32 accumulate reproduces the result.
// -----------------------------------------------------------------------------

struct WarpImma8816Ctx {
    std::mutex m;
    std::condition_variable cv;
    int      live       = 0;
    int      arrived    = 0;
    int      generation = 0;
    uint32_t a[WARP_SIZE];      // 1 packed-4xs8 A reg / lane
    uint32_t b[WARP_SIZE];      // 1 packed-4xs8 B reg / lane
    int32_t  c[WARP_SIZE][2];   // 2 C frags / lane
    int32_t  d[WARP_SIZE][2];   // 2 D frags / lane
};
static WarpImma8816Ctx* g_warpImma8816Ctx = nullptr;

extern "C" void imma_8816_s8_warp(int32_t* dout,
        int32_t aPacked, int32_t bPacked, int32_t c0, int32_t c1)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpImma8816Ctx& ctx = g_warpImma8816Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane] = static_cast<uint32_t>(aPacked);
    ctx.b[lane] = static_cast<uint32_t>(bPacked);
    ctx.c[lane][0] = c0; ctx.c[lane][1] = c1;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        int32_t A[8][16], B[8][16];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, tg = ln & 3;
            const uint32_t pa = ctx.a[ln], pb = ctx.b[ln];
            for (int v = 0; v < 4; ++v) {
                // sign-extend each packed byte to int8 then int32
                A[g][tg * 4 + v] = static_cast<int32_t>(
                    static_cast<int8_t>((pa >> (8 * v)) & 0xFFu));
                B[g][tg * 4 + v] = static_cast<int32_t>(
                    static_cast<int8_t>((pb >> (8 * v)) & 0xFFu));
            }
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, tg = ln & 3;
            for (int v = 0; v < 2; ++v) {
                const int m = g, n = tg * 2 + v;
                int32_t acc = ctx.c[ln][v];
                for (int k = 0; k < 16; ++k)
                    acc += A[m][k] * B[n][k];
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0] = ctx.d[lane][0]; dout[1] = ctx.d[lane][1];
}

// -----------------------------------------------------------------------------
// imma_16832_s8_warp — IMMA.16832 (m16n8k32, int8->int32): warp-true integer MMA.
// Matches IR:
//   declare void @"imma_16832_s8_warp"(int* %d, int a0, int a1, int a2, int a3,
//                                       int b0, int b1, int c0, int c1, int c2, int c3)
//
// The larger-shape sibling of imma_8816_s8_warp (the Ampere sm80 dense int8 atom;
// CUTLASS 13_two_tensor_op_fusion s8 variants issue 412-544 of these). The 16x8
// D tile (4 int32 vals/lane), the 16x32 A (4 packed-4xs8 regs/lane) and 8x32 B
// (2 packed-4xs8 regs/lane) int8 fragments are distributed across all 32 lanes.
// The earlier per-lane C->D identity stub dropped the A*B product entirely (every
// dense-s8 IMMA.16832 GEMM computed D = C). Same one-barrier rendezvous as
// imma_8816_s8_warp / dmma_884_f64: each lane parks its 4 A regs, 2 B regs and 4 C
// values; the last arriver unpacks + reconstructs the full int8 A[16][32] /
// B[8][32] and computes all lanes' int32 D, everyone reads back its own 4 values.
// Fragment->matrix mapping is the cute MMA_Traits<SM80_16x8x32_S32S8S8S32_TN>
// layout (ALayout = Shape<Shape<_4,_8>,Shape<_4,_2,_2>> Stride<Stride<_64,_1>,
// Stride<_16,_8,_256>>; BLayout = Shape<Shape<_4,_8>,Shape<_4,_2>> Stride<Stride<
// _32,_1>,Stride<_8,_128>>; CLayout = SM80_16x8_Row), decoded with cute's
// COLUMN-MAJOR (M-major) codomain — the SAME convention as IMMA.8816 / DMMA.8x8x4
// (verified all-once frag coverage A512/B256/C128 and a full m16n8k32 integer GEMM,
// maxdiff 0). lane = 4*g + tg (g = lane/4 in 0..7, tg = lane%4 in 0..3):
//   A: 4 regs, each 4 packed s8 (byte b at bits[8b..8b+7]):
//     r0: A[m=g  ][k=     tg*4+b]   r1: A[m=g+8][k=     tg*4+b]
//     r2: A[m=g  ][k=16 + tg*4+b]   r3: A[m=g+8][k=16 + tg*4+b]   (A is M16 x K32, row-major)
//   B: 2 regs, each 4 packed s8:
//     r0: B[n=g][k=     tg*4+b]     r1: B[n=g][k=16 + tg*4+b]     (B is N8 x K32, col-major K x N)
//   C/D: 4 int32 regs:
//     v0: C[m=g  ][n=tg*2+0]  v1: C[m=g  ][n=tg*2+1]
//     v2: C[m=g+8][n=tg*2+0]  v3: C[m=g+8][n=tg*2+1]              (C/D M16 x N8 int32)
//   D[m][n] = C[m][n] + sum_{k<32} A[m][k]*B[n][k]   (signed int8 products, s32 accum)
// int8 byte order little-endian (matches PTX __dp4a packing). Accumulation is exact
// int32 (no rounding); .satfinite cannot trip for these GEMM ranges, so a plain
// int32 accumulate reproduces the result.
// -----------------------------------------------------------------------------

struct WarpImma16832Ctx {
    std::mutex m;
    std::condition_variable cv;
    int      live       = 0;
    int      arrived    = 0;
    int      generation = 0;
    uint32_t a[WARP_SIZE][4];   // 4 packed-4xs8 A regs / lane
    uint32_t b[WARP_SIZE][2];   // 2 packed-4xs8 B regs / lane
    int32_t  c[WARP_SIZE][4];   // 4 C frags / lane
    int32_t  d[WARP_SIZE][4];   // 4 D frags / lane
};
static WarpImma16832Ctx* g_warpImma16832Ctx = nullptr;

extern "C" void imma_16832_s8_warp(int32_t* dout,
        int32_t a0, int32_t a1, int32_t a2, int32_t a3,
        int32_t b0, int32_t b1,
        int32_t c0, int32_t c1, int32_t c2, int32_t c3)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpImma16832Ctx& ctx = g_warpImma16832Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0] = static_cast<uint32_t>(a0); ctx.a[lane][1] = static_cast<uint32_t>(a1);
    ctx.a[lane][2] = static_cast<uint32_t>(a2); ctx.a[lane][3] = static_cast<uint32_t>(a3);
    ctx.b[lane][0] = static_cast<uint32_t>(b0); ctx.b[lane][1] = static_cast<uint32_t>(b1);
    ctx.c[lane][0] = c0; ctx.c[lane][1] = c1; ctx.c[lane][2] = c2; ctx.c[lane][3] = c3;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        int32_t A[16][32], B[8][32];
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, tg = ln & 3;
            // A: reg r in 0..3 -> (m = g + (r&1?8:0), k = (r&2?16:0) + tg*4 + b)
            for (int r = 0; r < 4; ++r) {
                const int m = g + ((r & 1) ? 8 : 0);
                const int kbase = ((r & 2) ? 16 : 0) + tg * 4;
                const uint32_t pa = ctx.a[ln][r];
                for (int b = 0; b < 4; ++b)
                    A[m][kbase + b] = static_cast<int32_t>(
                        static_cast<int8_t>((pa >> (8 * b)) & 0xFFu));
            }
            // B: reg r in 0..1 -> (n = g, k = (r?16:0) + tg*4 + b)
            for (int r = 0; r < 2; ++r) {
                const int n = g;
                const int kbase = (r ? 16 : 0) + tg * 4;
                const uint32_t pb = ctx.b[ln][r];
                for (int b = 0; b < 4; ++b)
                    B[n][kbase + b] = static_cast<int32_t>(
                        static_cast<int8_t>((pb >> (8 * b)) & 0xFFu));
            }
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int g = ln >> 2, tg = ln & 3;
            for (int v = 0; v < 4; ++v) {
                const int m = g + ((v >= 2) ? 8 : 0);
                const int n = tg * 2 + (v & 1);
                int32_t acc = ctx.c[ln][v];
                for (int k = 0; k < 32; ++k)
                    acc += A[m][k] * B[n][k];
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0] = ctx.d[lane][0]; dout[1] = ctx.d[lane][1];
    dout[2] = ctx.d[lane][2]; dout[3] = ctx.d[lane][3];
}

// -----------------------------------------------------------------------------
// imma_sp16864_s8_warp — IMMA.SP.16864 (m16n8k64, int8->int32, 2:4 structured
// sparse): warp-true integer sparse MMA.
// Matches IR:
//   declare void @"imma_sp16864_s8_warp"(int* %d, int a0,a1,a2,a3,
//          int b0,b1,b2,b3, int c0,c1,c2,c3, int e)
//
// CUTLASS 15_ampere_sparse_tensorop_gemm emits IMMA.SP.16864.S8.S8.SAT (PTX
// mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32.satfinite). A is SPARSE-
// COMPRESSED: only K/2 = 32 of every K=64 columns/row are stored; the metadata E
// (2 bits per group-of-4 K-cols, packed 4-bit fields) selects which 2 dense
// positions per group the kept values occupy. The earlier per-lane C->D identity
// stub dropped the A*B product entirely (D = C). Same one-barrier rendezvous as
// imma_8816_s8_warp: each lane parks its 4 packed A-comp regs, 4 packed B regs,
// 4 C, and 1 E; the last arriver reconstructs the full dense int8 A[16][64] (via
// the cutlass host_uncompress.h 2:4 rule) / B[8][64] and computes all lanes'
// int32 D, everyone reads back its own 4 values.
//
// Fragment maps (cute SM80_16x8x32_S32S8S8S32 strides, column/M-major codomain —
// SAME convention as the dense IMMA.8816 & DMMA fixes; all VERIFIED all-once in
// temp/imma_sparse/decode_sparse.py).  lane = t (0..31), t0 = t%4, t1 = t//4:
//   A-compressed (4 regs = 16 packed s8): for packed index p in 0..15:
//       off = 64*t0 + t1 + 16*(p%4) + 8*((p/4)%2) + 256*((p/8)%2);
//       m = off%16, ccol = off/16  (M16 x 32 compressed cols).
//     The 16 fragments form 8 PAIRS (pair j = frags {2j,2j+1}); each pair holds a
//     group g = ccol/2's two kept values for row m. E slot j (bits [4j..4j+3]) =
//     (idx0 | idx1<<2): dense pos 4g+idx0 <- frag 2j, 4g+idx1 <- frag 2j+1, rest 0.
//   B (4 regs = 16 packed s8): two stacked dense-k32 B atoms along K; for packed
//       index p: offB = 32*t0 + t1 + 8*((p%8)%4) + 128*((p%8)/4);
//       n = offB%8, k = offB/8 + 32*(p/8)  (N8 x K64, dense).
//   C/D (4 regs s32): m = t1 + 8*(v/2), n = t0*2 + (v%2)  v in 0..3.
//   D[m][n] = C[m][n] + sum_{k<64} A_dense[m][k]*B[n][k]  (signed int8 products,
//   s32 accum). HW .satfinite cannot trip for these GEMM ranges (CUTLASS picks
//   ranges so it never saturates), so a plain int32 accumulate reproduces it.
// int8 byte order in each 32-bit reg: byte v at bits [8v..8v+7] (little-endian).
// -----------------------------------------------------------------------------

struct WarpImmaSp16864Ctx {
    std::mutex m;
    std::condition_variable cv;
    int      live       = 0;
    int      arrived    = 0;
    int      generation = 0;
    uint32_t a[WARP_SIZE][4];   // 4 packed-4xs8 A-compressed regs / lane
    uint32_t b[WARP_SIZE][4];   // 4 packed-4xs8 B regs / lane
    int32_t  c[WARP_SIZE][4];   // 4 C frags / lane
    uint32_t e[WARP_SIZE];      // 1 metadata reg / lane
    int32_t  d[WARP_SIZE][4];   // 4 D frags / lane
};
static WarpImmaSp16864Ctx* g_warpImmaSp16864Ctx = nullptr;

extern "C" void imma_sp16864_s8_warp(int32_t* dout,
        int32_t a0, int32_t a1, int32_t a2, int32_t a3,
        int32_t b0, int32_t b1, int32_t b2, int32_t b3,
        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
        int32_t e)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpImmaSp16864Ctx& ctx = g_warpImmaSp16864Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0] = (uint32_t)a0; ctx.a[lane][1] = (uint32_t)a1;
    ctx.a[lane][2] = (uint32_t)a2; ctx.a[lane][3] = (uint32_t)a3;
    ctx.b[lane][0] = (uint32_t)b0; ctx.b[lane][1] = (uint32_t)b1;
    ctx.b[lane][2] = (uint32_t)b2; ctx.b[lane][3] = (uint32_t)b3;
    ctx.c[lane][0] = c0; ctx.c[lane][1] = c1;
    ctx.c[lane][2] = c2; ctx.c[lane][3] = c3;
    ctx.e[lane] = (uint32_t)e;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        int32_t A[16][64];   // dense reconstruction (zero-filled)
        int32_t B[8][64];
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 64; ++k) A[m][k] = 0;

        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            const uint32_t emeta = ctx.e[ln];
            // unpack this lane's 16 A-compressed s8 (4 regs) and place into A_dense
            // per the 2:4 metadata.  PAIR j = frags {2j, 2j+1}; slot j of E.
            for (int j = 0; j < 8; ++j) {
                const int p0 = 2 * j, p1 = 2 * j + 1;
                const int off = 64 * t0 + t1 + 16 * (p0 % 4) + 8 * ((p0 / 4) % 2) + 256 * ((p0 / 8) % 2);
                const int m = off % 16;
                const int g = (off / 16) / 2;       // dense group (ccol/2)
                const int field = (emeta >> (4 * j)) & 0xf;
                const int idx0 = field & 0x3;
                const int idx1 = (field >> 2) & 0x3;
                const int8_t v0 = (int8_t)((ctx.a[ln][p0 >> 2] >> (8 * (p0 & 3))) & 0xFFu);
                const int8_t v1 = (int8_t)((ctx.a[ln][p1 >> 2] >> (8 * (p1 & 3))) & 0xFFu);
                A[m][4 * g + idx0] = (int32_t)v0;
                A[m][4 * g + idx1] = (int32_t)v1;
            }
            // unpack this lane's 16 B s8 (4 regs) into dense B
            for (int p = 0; p < 16; ++p) {
                const int ps   = p % 8;
                const int offB = 32 * t0 + t1 + 8 * (ps % 4) + 128 * (ps / 4);
                const int bn   = offB % 8;
                const int bk   = (offB / 8) + 32 * (p / 8);
                B[bn][bk] = (int32_t)(int8_t)((ctx.b[ln][p >> 2] >> (8 * (p & 3))) & 0xFFu);
            }
        }
        // compute D = C + A_dense * B^T for every lane's 4 fragments
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            for (int v = 0; v < 4; ++v) {
                const int m = t1 + 8 * (v / 2);
                const int n = t0 * 2 + (v % 2);
                int32_t acc = ctx.c[ln][v];
                for (int k = 0; k < 64; ++k)
                    acc += A[m][k] * B[n][k];
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0] = ctx.d[lane][0]; dout[1] = ctx.d[lane][1];
    dout[2] = ctx.d[lane][2]; dout[3] = ctx.d[lane][3];
}

// -----------------------------------------------------------------------------
// imma_sp_s4_warp — INT4 m16n8k128 2:4-structured-sparse tensor-core MMA
// (the deferred follow-up to imma_sp16864_s8_warp). CUTLASS
// 15_ampere_sparse_tensorop_gemm is an integer_subbyte<4> (int4) 2:4-sparse GEMM
// whose warp atom is SparseMma<GemmShape<16,8,128>,32,int4b_t,...>. PTX:
//   mma.sp.sync.aligned.m16n8k128.row.col.s32.s4.s4.s32.satfinite
//      {D0..3}, {A0..3}, {B0..3}, {C0..3}, E, 0x0
// The SASS opcode is IMMA.SP.16864.S8.S8 (identical text to the int8 m16n8k64
// atom); the int4 dtype/shape are recovered from the kernel-symbol metadata and
// the lifter routes int4-tagged sparse atoms HERE instead of imma_sp16864_s8_warp.
// Matches IR (same arg shape as the int8 wrapper):
//   declare void @"imma_sp_s4_warp"(int* %d, int a0,a1,a2,a3, int b0,b1,b2,b3,
//          int c0,c1,c2,c3, int e)
//
// Fragments (cute SM80_16x8x64_S32S4S4S32 strides reused for the sparse operands;
// all VERIFIED all-once in temp/int4_sparse/decode_int4.py).  lane = t (0..31),
// t0 = t%4, t1 = t//4:
//   A-compressed (4 regs = 32 packed int4): for nibble index p in 0..31:
//       off = 128*t0 + t1 + 16*(p%8) + 8*((p/8)%2) + 512*((p/16)%2);
//       m = off%16, ccol = off/16  (M16 x 64 kept-cols).
//     The 32 fragments form 16 PAIRS (pair j = frags {2j,2j+1}); each is a kept
//     dense pair (int4 keeps in PAIRS: ElementsPerE=2). 16 pairs / lane form 8
//     nibble-GROUPS gg; comp pairs {2gg,2gg+1} are the 2 kept pairs of dense group
//     gg. E nibble gg = (idx0|idx1<<2): dense pair (4*?+idx0)<-comp pair 2gg,
//     dense pair (4*?+idx1)<-comp pair 2gg+1 (cutlass host_uncompress.h int4 rule:
//     step=8, a=4, b=8, ElementsPerE=2).
//   B (4 regs = 32 packed int4): two stacked dense-k64 atoms; for nibble index p:
//       ps=p%16; offB=64*t0+t1+8*(ps%8)+256*(ps/8); n=offB%8; k=offB/8 + 64*(p/16)
//       (N8 x K128, dense).
//   C/D (4 regs s32): m = t1 + 8*(v/2), n = t0*2 + (v%2)  v in 0..3.
//   D[m][n] = C[m][n] + sum_{k<128} A_dense[m][k]*B[n][k]  (signed int4 products,
//   s32 accum; .satfinite cannot trip for CUTLASS-chosen ranges).
// int4 nibble order in each 32-bit reg: nibble p at bits [4p..4p+3] (little-endian).
// -----------------------------------------------------------------------------

static inline int s4_sext(uint32_t nib) {           // sign-extend a 4-bit field
    int v = (int)(nib & 0xF);
    return (v & 0x8) ? (v - 16) : v;
}

struct WarpImmaSpS4Ctx {
    std::mutex m;
    std::condition_variable cv;
    int      live       = 0;
    int      arrived    = 0;
    int      generation = 0;
    uint32_t a[WARP_SIZE][4];   // 4 packed-8xs4 A-compressed regs / lane
    uint32_t b[WARP_SIZE][4];   // 4 packed-8xs4 B regs / lane
    int32_t  c[WARP_SIZE][4];   // 4 C frags / lane
    uint32_t e[WARP_SIZE];      // 1 metadata reg / lane
    int32_t  d[WARP_SIZE][4];   // 4 D frags / lane
};
static WarpImmaSpS4Ctx* g_warpImmaSpS4Ctx = nullptr;

extern "C" void imma_sp_s4_warp(int32_t* dout,
        int32_t a0, int32_t a1, int32_t a2, int32_t a3,
        int32_t b0, int32_t b1, int32_t b2, int32_t b3,
        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
        int32_t e)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpImmaSpS4Ctx& ctx = g_warpImmaSpS4Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0] = (uint32_t)a0; ctx.a[lane][1] = (uint32_t)a1;
    ctx.a[lane][2] = (uint32_t)a2; ctx.a[lane][3] = (uint32_t)a3;
    ctx.b[lane][0] = (uint32_t)b0; ctx.b[lane][1] = (uint32_t)b1;
    ctx.b[lane][2] = (uint32_t)b2; ctx.b[lane][3] = (uint32_t)b3;
    ctx.c[lane][0] = c0; ctx.c[lane][1] = c1;
    ctx.c[lane][2] = c2; ctx.c[lane][3] = c3;
    ctx.e[lane] = (uint32_t)e;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        int32_t A[16][128];   // dense reconstruction (zero-filled)
        int32_t B[8][128];
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 128; ++k) A[m][k] = 0;

        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            const uint32_t emeta = ctx.e[ln];
            // Unpack this lane's 32 A-compressed int4 (4 regs) into A_dense per the
            // int4 2:4 metadata.  PAIR j = frags {2j, 2j+1} (a kept dense pair).
            // 16 pairs / lane form 8 nibble-GROUPS gg (comp pairs {2gg,2gg+1} = the
            // 2 kept pairs of dense group gg). E nibble gg gives (idx0|idx1<<2)
            // selecting which 2 of the 4 dense pairs are non-zero.
            for (int gg = 0; gg < 8; ++gg) {
                const int field = (emeta >> (4 * gg)) & 0xf;
                const int idx0 = field & 0x3;
                const int idx1 = (field >> 2) & 0x3;
                for (int sel = 0; sel < 2; ++sel) {
                    const int j   = 2 * gg + sel;        // compressed pair index (0..15)
                    const int p0  = 2 * j, p1 = 2 * j + 1;   // the 2 nibbles of the pair
                    const int off = 128 * t0 + t1 + 16 * (p0 % 8) + 8 * ((p0 / 8) % 2) + 512 * ((p0 / 16) % 2);
                    const int m   = off % 16;
                    const int dg  = (off / 16) / 4;       // dense group (ccol/4) for this lane
                    const int idx = sel == 0 ? idx0 : idx1;
                    const int v0  = s4_sext((ctx.a[ln][p0 >> 3] >> (4 * (p0 & 7))) & 0xFu);
                    const int v1  = s4_sext((ctx.a[ln][p1 >> 3] >> (4 * (p1 & 7))) & 0xFu);
                    A[m][8 * dg + 2 * idx]     = v0;
                    A[m][8 * dg + 2 * idx + 1] = v1;
                }
            }
            // Unpack this lane's 32 B int4 (4 regs) into dense B (N8 x K128).
            for (int p = 0; p < 32; ++p) {
                const int ps   = p % 16;
                const int offB = 64 * t0 + t1 + 8 * (ps % 8) + 256 * (ps / 8);
                const int bn   = offB % 8;
                const int bk   = (offB / 8) + 64 * (p / 16);
                B[bn][bk] = s4_sext((ctx.b[ln][p >> 3] >> (4 * (p & 7))) & 0xFu);
            }
        }
        // compute D = C + A_dense * B^T for every lane's 4 fragments
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            for (int v = 0; v < 4; ++v) {
                const int m = t1 + 8 * (v / 2);
                const int n = t0 * 2 + (v % 2);
                int32_t acc = ctx.c[ln][v];
                for (int k = 0; k < 128; ++k)
                    acc += A[m][k] * B[n][k];
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0] = ctx.d[lane][0]; dout[1] = ctx.d[lane][1];
    dout[2] = ctx.d[lane][2]; dout[3] = ctx.d[lane][3];
}

// -----------------------------------------------------------------------------
// imma_sp_s4_k64_warp — NATIVE K=64 int4 sparse sub-atom of the int4 m16n8k128 atom.
//
// CUTLASS 15_ampere_sparse_tensorop_gemm's int4 warp atom is the LOGICAL
// SparseMma<GemmShape<16,8,128>,32,int4b_t,...> (PTX mma.sp.m16n8k128.s4.s4). nvcc
// lowers that ONE PTX op to a PAIR of native SASS IMMA.SP.16864.S8.S8 sub-atoms whose
// D the kernel SUMS (atom2.D += atom1.D) -> the K=128 result. The lifter routes EACH
// int4 sub-atom HERE (not the K=128 imma_sp_s4_warp): each native sub-atom is a true
// K=64 int4 sparse op, and the SASS sum yields K=128 — no pairing recognizer.
//
// WHY this is the int8-K64 thread map reading int4 low-nibbles (DERIVED + verified
// 32/32 lanes on this host, no GPU; temp/int4_k64/{derive,deriveB,parity_free}.py):
//   ptxas splits the logical fragments by NIBBLE PARITY: sub-atom0 gets the EVEN
//   logical nibbles, sub-atom1 the ODD — each packed as the LOW NIBBLE of the 16 bytes
//   of its 4 A-regs / 4 B-regs (high nibble is junk). Feeding a sub-atom's 16 A-comp +
//   16 B int4 (low-nibble-of-byte) through the PROVEN int8 imma_sp16864_s8_warp K=64
//   thread map reconstructs a dense A[16][64]/B[8][64] whose K=64 GEMM is exactly HALF
//   the logical K=128 product; summing the two sub-atoms (the SASS IMAD.IADD) gives the
//   full result. *** PARITY-AGNOSTIC: identical code for both sub-atoms *** — the data
//   placement (which nibbles ptxas routed to this sub-atom's regs) encodes the K-window.
//   Validated: SUM(subatom0)+SUM(subatom1)+C == dense int4 oracle, bad=0/128, on two
//   independent high-entropy probes.
//
// Same arg shape as imma_sp_s4_warp (1 D-ptr + 4 A + 4 B + 4 C + 1 E i32). Reuses the
// WarpImmaSpS4Ctx storage (mutually exclusive with imma_sp_s4_warp within a kernel).
// int4 read: low nibble of byte v in reg r = s4_sext((r >> (8v)) & 0xF).
// -----------------------------------------------------------------------------

extern "C" void imma_sp_s4_k64_warp(int32_t* dout,
        int32_t a0, int32_t a1, int32_t a2, int32_t a3,
        int32_t b0, int32_t b1, int32_t b2, int32_t b3,
        int32_t c0, int32_t c1, int32_t c2, int32_t c3,
        int32_t e)
{
    const int lane       = g_threadCtx.lane;
    const int warpGlobal = g_threadCtx.warpGlobal;
    WarpImmaSpS4Ctx& ctx = g_warpImmaSpS4Ctx[warpGlobal];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    ctx.a[lane][0] = (uint32_t)a0; ctx.a[lane][1] = (uint32_t)a1;
    ctx.a[lane][2] = (uint32_t)a2; ctx.a[lane][3] = (uint32_t)a3;
    ctx.b[lane][0] = (uint32_t)b0; ctx.b[lane][1] = (uint32_t)b1;
    ctx.b[lane][2] = (uint32_t)b2; ctx.b[lane][3] = (uint32_t)b3;
    ctx.c[lane][0] = c0; ctx.c[lane][1] = c1;
    ctx.c[lane][2] = c2; ctx.c[lane][3] = c3;
    ctx.e[lane] = (uint32_t)e;
    ctx.arrived++;

    warpCollectiveComplete(ctx, lock, myGen, [&] {
        int32_t A[16][64];   // dense reconstruction over the sub-atom's K=64 window
        int32_t B[8][64];
        for (int m = 0; m < 16; ++m)
            for (int k = 0; k < 64; ++k) A[m][k] = 0;

        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            const uint32_t emeta = ctx.e[ln];
            // A: 16 int4 A-comp (low nibble of each of 16 bytes across 4 regs) form 8
            // PAIRS (pair j = bytes {2j, 2j+1}); E slot j of the metadata selects the
            // 2 kept dense positions of group g = ccol/2. (int8 imma_sp16864 K=64 map.)
            for (int j = 0; j < 8; ++j) {
                const int p0 = 2 * j, p1 = 2 * j + 1;
                const int off = 64 * t0 + t1 + 16 * (p0 % 4) + 8 * ((p0 / 4) % 2) + 256 * ((p0 / 8) % 2);
                const int m = off % 16;
                const int g = (off / 16) / 2;       // dense group (ccol/2)
                const int field = (emeta >> (4 * j)) & 0xf;
                const int idx0 = field & 0x3;
                const int idx1 = (field >> 2) & 0x3;
                // int4 value = low nibble of the byte (high nibble is ptxas junk).
                const int v0 = s4_sext((ctx.a[ln][p0 >> 2] >> (8 * (p0 & 3))) & 0xFu);
                const int v1 = s4_sext((ctx.a[ln][p1 >> 2] >> (8 * (p1 & 3))) & 0xFu);
                A[m][4 * g + idx0] = v0;
                A[m][4 * g + idx1] = v1;
            }
            // B: 16 int4 (low nibble of each byte) dense N8 x K64 (int8 K=64 B map).
            for (int p = 0; p < 16; ++p) {
                const int ps   = p % 8;
                const int offB = 32 * t0 + t1 + 8 * (ps % 4) + 128 * (ps / 4);
                const int bn   = offB % 8;
                const int bk   = (offB / 8) + 32 * (p / 8);
                B[bn][bk] = s4_sext((ctx.b[ln][p >> 2] >> (8 * (p & 3))) & 0xFu);
            }
        }
        for (int ln = 0; ln < WARP_SIZE; ++ln) {
            const int t0 = ln & 3, t1 = ln >> 2;
            for (int v = 0; v < 4; ++v) {
                const int m = t1 + 8 * (v / 2);
                const int n = t0 * 2 + (v % 2);
                int32_t acc = ctx.c[ln][v];
                for (int k = 0; k < 64; ++k)
                    acc += A[m][k] * B[n][k];
                ctx.d[ln][v] = acc;
            }
        }
    });

    dout[0] = ctx.d[lane][0]; dout[1] = ctx.d[lane][1];
    dout[2] = ctx.d[lane][2]; dout[3] = ctx.d[lane][3];
}

// -----------------------------------------------------------------------------
// vote_uni
// Matches IR:
//   declare i8 @"vote_uni"(i32 %mask, i8 %pred, i32 %site)
// Returns true if all active lanes have the same predicate value.
// -----------------------------------------------------------------------------

extern "C" bool vote_uni(uint32_t mask, bool pred, uint32_t site)
{
    (void)mask;
    const int warpStartTid = g_threadCtx.warpInBlock * WARP_SIZE;
    const int lanesInWarp  = std::min(WARP_SIZE, g_blockDim - warpStartTid);
    if (lanesInWarp <= 1) return pred;
    uint32_t vals[WARP_SIZE]; bool preds[WARP_SIZE] = {};
    const uint32_t members = warp_group_gather(site, 0, pred, vals, preds);
    uint32_t grp_ballot = 0;
    for (int M = 0; M < WARP_SIZE; ++M)
        if (((members >> M) & 1) && preds[M]) grp_ballot |= (1u << M);
    return grp_ballot == 0 || grp_ballot == members;
}

// BAR.RED.{AND,OR,POPC}: block-wide barrier 0 + predicate reduction.
// Implemented after syncthreads (shares BlockSyncCtx barrier-0 state);
// see barrier0_red_. An identity stub here is NOT safe: CUTLASS
// Semaphore::wait() spins on `__syncthreads_and(state != expected)` where
// only tid 0 ever refreshes `state` — every other thread needs tid 0's
// predicate broadcast to exit (58_ada_fp8_gemm infinite spin).

// -----------------------------------------------------------------------------
// syncthreads
// Matches IR:
//   declare void @"syncthreads"()
// -----------------------------------------------------------------------------

// Clear every parked flag in `cta`'s warps. MUST be called by the lane that
// releases a block barrier, while it still holds the barrier mutex (all
// released lanes are then provably blocked in cv.wait): a releasee sprinting
// into the next collective must never see a peer's stale parked=true — that
// classifies the peer "not coming" and drops its value from the activemask
// (observed: two-stage block reductions losing one warp partial per run).
static void clear_cta_parked_(int cta)
{
    if (g_warpHistCtx) {
        for (int w = 0; w < g_warpsPerBlock; ++w) {
            const int wg = cta * g_warpsPerBlock + w;
            { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg].m);
              for (int l = 0; l < WARP_SIZE; ++l) g_warpHistCtx[wg].parked[l] = false; }
            if (g_warpGroupCtx) g_warpGroupCtx[wg].cv.notify_all();
        }
    }
    // Lockstep turnstile: unpark all lanes HERE, not lane-by-lane in each
    // TurnParkGuard dtor after wake-up. Dtors run in arbitrary scheduling
    // order, so the token could traverse while a slow waker is still marked
    // parked — skipping it and costing it its slot in the first post-barrier
    // round (observed: C_scan4 warp-prefix corruption ~1/5 runs). At release
    // every CTA thread is at the barrier, so no lane is parked in a warp
    // collective — mass-clearing is safe. The dtor's own unpark/reclaim then
    // degenerates to a no-op.
    if (g_warpTurnCtx) {
        for (int w = 0; w < g_warpsPerBlock; ++w) {
            const int wg = cta * g_warpsPerBlock + w;
            WarpTurnCtx& c = g_warpTurnCtx[wg];
            { std::lock_guard<std::mutex> lk(c.m);
              for (int l = 0; l < c.lanes; ++l) c.parked[l] = false;
              if (c.exited[c.cur]) warp_turn_advance_(c); }
            c.cv.notify_all();
        }
    }
}

// Complete block barrier 0: snapshot + reset the BAR.RED reduction state,
// flip the generation, and clear parked flags. Caller holds ctx.m. Shared
// by syncthreads, barrier0_red_, and the thread-exit path so that mixed
// BAR.SYNC / BAR.RED / EXIT arrivals complete one coherent barrier.
static void block_barrier_complete_(BlockSyncCtx& ctx, int cta)
{
    ctx.red_and_res  = ctx.red_and;
    ctx.red_or_res   = ctx.red_or;
    ctx.red_popc_res = ctx.red_popc;
    ctx.red_and = 1; ctx.red_or = 0; ctx.red_popc = 0;
    ctx.arrived = 0;
    ++ctx.generation;
    clear_cta_parked_(cta);
}

extern "C" void syncthreads()
{
    if (const uint64_t n_ = spin_hotcall("syncthreads"))
        std::fprintf(stderr, "HOTCALL site=syncthreads tid=%d cta=%d count=%llu\n",
                     g_threadCtx.tid, g_threadCtx.cta, (unsigned long long)n_);
    // Lockstep mode: leave the warp's turn rotation while at the block
    // barrier (the token must not wait on a lane that is waiting here).
    const int cta = g_threadCtx.cta;
    // Serialized-warp fiber path: single OS thread, so block by YIELDING (a
    // condvar wait would freeze every lane). Generation-based reuse: the last
    // arriver flips the generation; earlier arrivers yield until it changes.
    if (warp_fiber_active()) {
        BlockSyncCtx& ctx = g_blockSyncCtx[cta];
        const int myGen = ctx.generation;
        if (++ctx.arrived >= ctx.live) { ctx.arrived = 0; ++ctx.generation; }
        else { while (ctx.generation == myGen) g_warp_yield(); }
        return;
    }
    // Lockstep mode: leave the warp's turn rotation while at the block
    // barrier (the token must not wait on a lane that is waiting here).
    TurnParkGuard turn_park_;
    // Mark parked for the warp-collective machinery (and wake its waiters):
    // a lane waiting at a block barrier is not in any activemask.
    const int wg_ = g_threadCtx.warpGlobal;
    if (g_warpHistCtx) {
        { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg_].m);
          g_warpHistCtx[wg_].parked[g_threadCtx.lane] = true; }
        if (g_warpGroupCtx) g_warpGroupCtx[wg_].cv.notify_all();
    }

    BlockSyncCtx& ctx = g_blockSyncCtx[cta];

    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;

    ctx.arrived++;

    // Barrier completes against the LIVE thread count, not blockDim:
    // exited threads no longer participate (HW EXIT semantics).
    if (ctx.arrived >= ctx.live) {
        block_barrier_complete_(ctx, cta);
        lock.unlock();
        ctx.cv.notify_all();
    } else {
        spin_diag_wait(ctx.cv, lock, [&]{ return ctx.generation != myGen; }, [&]{
            std::fprintf(stderr, "SPIN site=syncthreads tid=%d cta=%d arrived=%d "
                         "live=%d gen=%d mygen=%d\n", g_threadCtx.tid, cta,
                         ctx.arrived, ctx.live, ctx.generation, myGen);
        });
    }
    // Far side of the barrier: the warp is reconverged. Resync the per-lane
    // divergence history to a warp-uniform epoch (post-barrier generation) so a
    // subsequent full-warp collective sees every lane (see
    // warp_hist_resync_after_barrier). Done outside ctx.m: g_laneHist is TLS and
    // warp_hist_publish takes its own per-warp lock.
    warp_hist_resync_after_barrier((uint64_t)(myGen + 1));
}

// BAR.RED.{AND,OR,POPC}.PRED — arrive at block barrier 0 contributing
// `pred` to a CTA-wide reduction; all arrivers receive the reduced value
// once every live thread has arrived. Declared by the x86 lifter as
//   declare i32 @barrier0_and(i32) / @barrier0_or / @barrier0_popc.
// Shares arrived/generation/live with syncthreads (HW: same barrier 0).
static int32_t barrier0_red_(int mode /*0=and 1=or 2=popc*/, int32_t pred)
{
    if (!g_blockSyncCtx)
        return mode == 2 ? (pred ? 1 : 0) : pred;  // pre-launch / 1-thread fallback
    if (const uint64_t n_ = spin_hotcall("barrier0_red"))
        std::fprintf(stderr, "HOTCALL site=barrier0_red mode=%d tid=%d cta=%d "
                     "pred=%d count=%llu\n", mode, g_threadCtx.tid, g_threadCtx.cta,
                     (int)pred, (unsigned long long)n_);
    TurnParkGuard turn_park_;
    const int cta = g_threadCtx.cta;
    // Mark parked for the warp-collective machinery (same as syncthreads):
    // a lane waiting at a block barrier is not in any activemask.
    const int wg_ = g_threadCtx.warpGlobal;
    if (g_warpHistCtx) {
        { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg_].m);
          g_warpHistCtx[wg_].parked[g_threadCtx.lane] = true; }
        if (g_warpGroupCtx) g_warpGroupCtx[wg_].cv.notify_all();
    }
    BlockSyncCtx& ctx = g_blockSyncCtx[cta];
    std::unique_lock<std::mutex> lock(ctx.m);
    const int myGen = ctx.generation;
    if (pred) { ctx.red_or = 1; ctx.red_popc++; }
    else      { ctx.red_and = 0; }
    ctx.arrived++;
    if (ctx.arrived >= ctx.live) {
        block_barrier_complete_(ctx, cta);
        const int32_t r = (mode == 0) ? ctx.red_and_res
                        : (mode == 1) ? ctx.red_or_res
                                      : ctx.red_popc_res;
        lock.unlock();
        ctx.cv.notify_all();
        // Block barrier = full reconvergence point (see syncthreads).
        warp_hist_resync_after_barrier((uint64_t)(myGen + 1));
        return r;
    }
    spin_diag_wait(ctx.cv, lock, [&]{ return ctx.generation != myGen; }, [&]{
        std::fprintf(stderr, "SPIN site=barrier0_red mode=%d tid=%d cta=%d pred=%d "
                     "arrived=%d live=%d gen=%d mygen=%d\n", mode, g_threadCtx.tid,
                     cta, (int)pred, ctx.arrived, ctx.live, ctx.generation, myGen);
    });
    const int32_t r = (mode == 0) ? ctx.red_and_res
                    : (mode == 1) ? ctx.red_or_res
                                  : ctx.red_popc_res;
    lock.unlock();
    // Block barrier = full reconvergence point (see syncthreads). Resync outside
    // ctx.m so the warp-history lock is never taken under the block-sync lock.
    warp_hist_resync_after_barrier((uint64_t)(myGen + 1));
    return r;
}
extern "C" int32_t barrier0_and(int32_t pred)  { return barrier0_red_(0, pred); }
extern "C" int32_t barrier0_or(int32_t pred)   { return barrier0_red_(1, pred); }
extern "C" int32_t barrier0_popc(int32_t pred) { return barrier0_red_(2, pred); }

// -----------------------------------------------------------------------------
// Generic address-space queries (QSPC / PTX isspacep)
// Our memory model: shared and local accesses are lifted as small OFFSETS
// into the wrapper arrays (shared_mem 228KB payload + 16KB guard, local_mem
// 32KB); global addresses are host heap/stack pointers (far above 1MB on
// Linux). The queries classify by magnitude — exact for every address the
// lifted code can actually produce. Guard-range offsets still classify as
// shared: touching them faults into the SKIP_SHMEM handler, which is the
// correct verdict for an over-carveout config.
// -----------------------------------------------------------------------------

extern "C" bool is_shared_address(uint64_t a) { return a < sizeof(shared_mem); }
extern "C" bool is_local_address(uint64_t a)  { return a < sizeof(local_mem); }
extern "C" bool is_global_address(uint64_t a) { return a >= (1ull << 20); }
extern "C" bool is_const_address(uint64_t a)  { (void)a; return false; }

// -----------------------------------------------------------------------------
// __assertfail
// Device-side assert() handler (FlashAttention cubins import it via SASS CALL).
// Exact CUDA device ABI:
//   void __assertfail(const char *message, const char *file, unsigned line,
//                     const char *function, size_t charSize)
// charSize distinguishes char/wchar_t messages; only char reaches us here.
//
// Note: device printf (`vprintf`) cannot live in this header — the device
// version takes a CUDA packed-argument buffer (void*), which collides with
// <cstdio>'s va_list declaration in any C++ TU. It lives in printf_runtime.c
// (plain C, compiled separately) instead.
// -----------------------------------------------------------------------------

extern "C" void __assertfail(const char* message, const char* file,
                             unsigned line, const char* function,
                             size_t /*charSize*/)
{
    std::fprintf(stderr, "%s:%u: %s: device assertion failed: %s\n",
                 file ? file : "?", line, function ? function : "?",
                 message ? message : "?");
    // No abort: on HW a failed device assert traps the KERNEL, but the host
    // process continues (cudaDeviceSynchronize returns cudaErrorAssert).
    // assert-cuda deliberately fires asserts for out-of-range threads and
    // expects the app to keep running; aborting turned its pass into a
    // process crash.
}

// -----------------------------------------------------------------------------
// Kernel launcher
//
// Interface:
//   template<typename Func, typename... Args>
//   void launchKernel(Func func, int gridDim, int blockDim, Args... args)
//
// Important: we give *per-block* shared memory semantics by running CTAs
// sequentially and zeroing shared_mem once per CTA, not per thread.
// -----------------------------------------------------------------------------

template<typename Func, typename... Args>
void launchKernel(Func func, int gridDim, int blockDim, Args... args)
{
    off = ARG_BASE;

    // Marshal arguments into this (main) thread's const_mem (c[0])
    (writeArg(args), ...);

    // Snapshot entire const_mem (all 5 banks) as a template
    std::memcpy(g_constMemTemplate.data(), const_mem, sizeof(const_mem));

    // Global config
    g_blockDim      = blockDim;
    g_gridDim       = gridDim;
    g_warpsPerBlock = (blockDim + WARP_SIZE - 1) / WARP_SIZE;
    const int totalWarps = g_warpsPerBlock * gridDim;

    // Lockstep turnstile (opt-in; pairs with IR lifted under
    // CULIFTER_LOCKSTEP=1 — without those begin/end calls arming is inert)
    std::vector<WarpTurnCtx> turnCtx;
    if (std::getenv("SLIFTER_LOCKSTEP")) {
        turnCtx = std::vector<WarpTurnCtx>(static_cast<size_t>(totalWarps));
        g_warpTurnCtx = turnCtx.data();
    }

    // Per-warp contexts (enough for all warps in the grid)
    std::vector<WarpShuffleCtx>  warpCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteCtx>     voteCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteAllCtx>  voteAllCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpHmmaCtx>     hmmaCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpLdsmCtx>     ldsmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpStsmCtx>     stsmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpDmmaCtx>     dmmaCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpDmma884Ctx>  dmma884Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImma8816Ctx> imma8816Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImma16832Ctx> imma16832Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImmaSp16864Ctx> immaSp16864Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImmaSpS4Ctx> immaSpS4Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpMovmCtx>     movmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpHistCtx>     histCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpGroupCtx>    groupCtx(static_cast<size_t>(totalWarps));
    g_warpCtx        = warpCtx.data();
    g_warpVoteCtx    = voteCtx.data();
    g_warpVoteAllCtx = voteAllCtx.data();
    g_warpHmmaCtx    = hmmaCtx.data();
    g_warpLdsmCtx    = ldsmCtx.data();
    g_warpStsmCtx    = stsmCtx.data();
    g_warpDmmaCtx    = dmmaCtx.data();
    g_warpDmma884Ctx = dmma884Ctx.data();
    g_warpImma8816Ctx = imma8816Ctx.data();
    g_warpImma16832Ctx = imma16832Ctx.data();
    g_warpImmaSp16864Ctx = immaSp16864Ctx.data();
    g_warpImmaSpS4Ctx = immaSpS4Ctx.data();
    g_warpMovmCtx    = movmCtx.data();
    g_warpHistCtx    = histCtx.data();
    g_warpGroupCtx   = groupCtx.data();

    // Per-block barrier contexts (one per CTA)
    std::vector<BlockSyncCtx> blockCtx(static_cast<size_t>(gridDim));
    g_blockSyncCtx = blockCtx.data();

    // -------------------------------------------------------------------------
    // OPT-IN CONCURRENT-CTA MODE (CULIFTER_CONCURRENT_CTA)
    //
    // The default below runs CTAs SEQUENTIALLY — correct for the overwhelming
    // majority of kernels (each CTA independent / phase-separated by a kernel
    // boundary), and the ONLY model under which the single global @shared_mem is
    // safe. But an in-kernel CROSS-CTA spin-barrier (stream-K's partial-tile
    // fixup: REDG atomic-reduce + ATOMG.CAS spin + GPU fence) DEADLOCKS under it:
    // a CTA spinning on a global flag a PEER CTA must set waits forever, because
    // the peer's threads have not been spawned yet (see
    // test/sm90_synth/cross_cta_spin).
    //
    // When CULIFTER_CONCURRENT_CTA is set, spawn ALL gridDim*blockDim threads at
    // once so peer CTAs make forward progress concurrently and the spin resolves.
    // The default (flag unset) path is byte-identical (this whole block is skipped).
    //
    // SAFETY RAIL: this does NOT change @shared_mem (still ONE global). It is
    // therefore correct ONLY for kernels whose shared memory is not used
    // concurrently across CTAs (stream-K's cross-CTA state is all GLOBAL — the
    // partials + barrier workspace; its shared mem is intra-CTA mainloop scratch).
    // A general rollout must add PER-CTA shared segments (see the impl plan in
    // below). Cross-CTA __syncthreads/shared collectives
    // are NOT modeled here.
    // -------------------------------------------------------------------------
    if (std::getenv("CULIFTER_CONCURRENT_CTA")) {
        static const uint64_t initialRingCC[1] = {0xcbf29ce484222325ull};
        // Pre-initialize ALL CTAs' per-warp/block contexts BEFORE any thread runs
        // (the ctx arrays are already sized for the whole grid).
        for (int cta = 0; cta < gridDim; ++cta) {
            blockCtx[static_cast<size_t>(cta)].live = blockDim;
            for (int w = 0; w < g_warpsPerBlock; ++w) {
                const int wg = cta * g_warpsPerBlock + w;
                const int lanes = std::min(WARP_SIZE, blockDim - w * WARP_SIZE);
                warpCtx[static_cast<size_t>(wg)].live    = lanes;
                voteCtx[static_cast<size_t>(wg)].live    = lanes;
                voteAllCtx[static_cast<size_t>(wg)].live = lanes;
                hmmaCtx[static_cast<size_t>(wg)].live    = lanes;
                ldsmCtx[static_cast<size_t>(wg)].live    = lanes;
                stsmCtx[static_cast<size_t>(wg)].live    = lanes;
                dmmaCtx[static_cast<size_t>(wg)].live    = lanes;
                dmma884Ctx[static_cast<size_t>(wg)].live = lanes;
                imma8816Ctx[static_cast<size_t>(wg)].live = lanes;
                imma16832Ctx[static_cast<size_t>(wg)].live = lanes;
                immaSp16864Ctx[static_cast<size_t>(wg)].live = lanes;
                immaSpS4Ctx[static_cast<size_t>(wg)].live = lanes;
                movmCtx[static_cast<size_t>(wg)].live    = lanes;
                if (g_warpTurnCtx) turnCtx[static_cast<size_t>(wg)].lanes = lanes;
                for (int l = 0; l < lanes; ++l) {
                    histCtx[static_cast<size_t>(wg)].seq[l]  = 0;
                    histCtx[static_cast<size_t>(wg)].hash[l] = 0xcbf29ce484222325ull;
                    histCtx[static_cast<size_t>(wg)].ring[l] = initialRingCC;
                }
            }
        }
        // NOTE: @shared_mem is zeroed ONCE (not per-CTA). With concurrent CTAs
        // there is no per-CTA reset point; supported only for shared-mem-free /
        // non-cross-CTA-shared kernels (see safety rail above).
        std::memset(shared_mem, 0, SLIFTER_SHM_PAYLOAD_BYTES);

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(gridDim) * static_cast<size_t>(blockDim));
        for (int cta = 0; cta < gridDim; ++cta) {
            for (int tid = 0; tid < blockDim; ++tid) {
                threads.emplace_back([=]() {
                    std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));
                    std::memset(local_mem, 0, sizeof(local_mem));
                    uint8_t* c0 = cmem0();
                    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X)     = static_cast<uint32_t>(blockDim);
                    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Y)     = 1u;
                    *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Z)     = 1u;
                    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = static_cast<uint32_t>(gridDim);
                    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Y) = 1u;
                    *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Z) = 1u;
                    *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X)    = static_cast<uint32_t>(cta);
                    *reinterpret_cast<uint32_t*>(c0 + SR_TID_X)      = static_cast<uint32_t>(tid);
                    *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID)    = static_cast<uint32_t>(tid & (WARP_SIZE - 1));
                    g_threadCtx.cta         = cta;
                    g_threadCtx.tid         = tid;
                    g_threadCtx.lane        = tid & (WARP_SIZE - 1);
                    g_threadCtx.warpInBlock = tid / WARP_SIZE;
                    g_threadCtx.warpGlobal  = cta * g_warpsPerBlock + g_threadCtx.warpInBlock;
                    g_laneHist = LaneHistory{};
                    warp_hist_publish();

                    func();

                    warp_turn_exit_();
                    {
                        const int wg2 = g_threadCtx.warpGlobal;
                        { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg2].m);
                          g_warpHistCtx[wg2].ring[g_threadCtx.lane] = nullptr; }
                        g_warpGroupCtx[wg2].cv.notify_all();
                    }
                    {
                        const int wg = g_threadCtx.warpGlobal;
                        { std::lock_guard<std::mutex> lk(g_warpCtx[wg].m);        g_warpCtx[wg].live--; }
                        g_warpCtx[wg].cv.notify_all();
                        { std::lock_guard<std::mutex> lk(g_warpVoteCtx[wg].m);    g_warpVoteCtx[wg].live--; }
                        g_warpVoteCtx[wg].cv.notify_all();
                        { std::lock_guard<std::mutex> lk(g_warpVoteAllCtx[wg].m); g_warpVoteAllCtx[wg].live--; }
                        g_warpVoteAllCtx[wg].cv.notify_all();
                    }
                    {
                        BlockSyncCtx& bctx = g_blockSyncCtx[cta];
                        std::unique_lock<std::mutex> lk(bctx.m);
                        bctx.live--;
                        if (bctx.arrived >= bctx.live && bctx.arrived > 0) {
                            block_barrier_complete_(bctx, cta);
                            lk.unlock();
                            bctx.cv.notify_all();
                        }
                    }
                });
            }
        }
        for (auto& t : threads) t.join();

        g_warpCtx        = nullptr;
        g_warpVoteCtx    = nullptr;
        g_warpVoteAllCtx = nullptr;
        g_blockSyncCtx   = nullptr;
        g_warpHistCtx    = nullptr;
        g_warpGroupCtx   = nullptr;
        g_warpTurnCtx    = nullptr;
        return;
    }

    // Run CTAs sequentially to give each one a fresh shared_mem
    for (int cta = 0; cta < gridDim; ++cta) {
        // Per-CTA shared memory init (NOT per thread)
        std::memset(shared_mem, 0, SLIFTER_SHM_PAYLOAD_BYTES);  // payload only — tail is the PROT_NONE guard

        blockCtx[static_cast<size_t>(cta)].live = blockDim;
        for (int w = 0; w < g_warpsPerBlock; ++w) {
            const int wg = cta * g_warpsPerBlock + w;
            const int lanes = std::min(WARP_SIZE, blockDim - w * WARP_SIZE);
            warpCtx[static_cast<size_t>(wg)].live    = lanes;
            voteCtx[static_cast<size_t>(wg)].live    = lanes;
            voteAllCtx[static_cast<size_t>(wg)].live = lanes;
            hmmaCtx[static_cast<size_t>(wg)].live    = lanes;
            ldsmCtx[static_cast<size_t>(wg)].live    = lanes;
            stsmCtx[static_cast<size_t>(wg)].live    = lanes;
            dmmaCtx[static_cast<size_t>(wg)].live    = lanes;
            dmma884Ctx[static_cast<size_t>(wg)].live = lanes;
            imma8816Ctx[static_cast<size_t>(wg)].live = lanes;
            imma16832Ctx[static_cast<size_t>(wg)].live = lanes;
            immaSp16864Ctx[static_cast<size_t>(wg)].live = lanes;
            immaSpS4Ctx[static_cast<size_t>(wg)].live = lanes;
            movmCtx[static_cast<size_t>(wg)].live    = lanes;
            if (g_warpTurnCtx) turnCtx[static_cast<size_t>(wg)].lanes = lanes;
            // Pre-publish each to-be-spawned lane's initial history NOW:
            // a null ring means "exited", and threads are spawned (and run)
            // sequentially — without this, the first lane races through its
            // first collective seeing unspawned peers as exited (singleton
            // groups). The dummy ring is replaced by the thread's TLS ring
            // at its first publish; only the pointer's non-null-ness and
            // (seq=0, hash=FNV basis) are read before then.
            static const uint64_t initialRing[1] = {0xcbf29ce484222325ull};
            for (int l = 0; l < lanes; ++l) {
                histCtx[static_cast<size_t>(wg)].seq[l]  = 0;
                histCtx[static_cast<size_t>(wg)].hash[l] = 0xcbf29ce484222325ull;
                histCtx[static_cast<size_t>(wg)].ring[l] = initialRing;
            }
        }

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(blockDim));

        for (int tid = 0; tid < blockDim; ++tid) {
            threads.emplace_back([=]() {
                // Initialize this thread's TLS const_mem from template
                std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));

                // local_mem is thread-local; zero if you want deterministic state
                std::memset(local_mem, 0, sizeof(local_mem));

                // Set special registers in c[0]
                uint8_t* c0 = cmem0();
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X)     = static_cast<uint32_t>(blockDim);
                // CUDA guarantees unspecified dims read 1, not 0 — a zero
                // here turns gridDim.y-based strides into infinite loops
                // (wedford's welford grid-stride spin).
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Y)     = 1u;
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Z)     = 1u;
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = static_cast<uint32_t>(gridDim);
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Y) = 1u;
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Z) = 1u;
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X)    = static_cast<uint32_t>(cta);
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_X)      = static_cast<uint32_t>(tid);
                *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID)    = static_cast<uint32_t>(tid & (WARP_SIZE - 1));

                // Per-thread context for warp primitives & syncthreads
                g_threadCtx.cta         = cta;
                g_threadCtx.tid         = tid;
                g_threadCtx.lane        = tid & (WARP_SIZE - 1);
                g_threadCtx.warpInBlock = tid / WARP_SIZE;
                g_threadCtx.warpGlobal  = cta * g_warpsPerBlock + g_threadCtx.warpInBlock;

                // Fresh divergence history for this logical thread; publish
                // the initial state so group collectives can classify us.
                g_laneHist = LaneHistory{};
                warp_hist_publish();

                // Run the lifted kernel for this logical CUDA thread
                func();

                // Lockstep turnstile: lane leaves the token rotation.
                warp_turn_exit_();

                // Thread exits: drop out of barrier participation and, if
                // it was the last straggler others were waiting on, release
                // the barrier (HW EXIT semantics).
                // History/group machinery: mark exited (ring=nullptr) and wake
                // group waiters — an exited lane can never join an instance.
                {
                    const int wg2 = g_threadCtx.warpGlobal;
                    { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg2].m);
                      g_warpHistCtx[wg2].ring[g_threadCtx.lane] = nullptr; }
                    g_warpGroupCtx[wg2].cv.notify_all();
                }
                // Warp collectives: this lane leaves the activemask. A
                // pending collective whose remaining live lanes have all
                // arrived is completed by the first waiter that wakes
                // (warpCollectiveComplete's wait predicate).
                {
                    const int wg = g_threadCtx.warpGlobal;
                    { std::lock_guard<std::mutex> lk(g_warpCtx[wg].m);        g_warpCtx[wg].live--; }
                    g_warpCtx[wg].cv.notify_all();
                    { std::lock_guard<std::mutex> lk(g_warpVoteCtx[wg].m);    g_warpVoteCtx[wg].live--; }
                    g_warpVoteCtx[wg].cv.notify_all();
                    { std::lock_guard<std::mutex> lk(g_warpVoteAllCtx[wg].m); g_warpVoteAllCtx[wg].live--; }
                    g_warpVoteAllCtx[wg].cv.notify_all();
                }
                {
                    BlockSyncCtx& bctx = g_blockSyncCtx[cta];
                    std::unique_lock<std::mutex> lk(bctx.m);
                    bctx.live--;
                    if (bctx.arrived >= bctx.live && bctx.arrived > 0) {
                        // Snapshot BAR.RED state too: pending barrier0_*
                        // waiters reduce over the threads that DID arrive.
                        block_barrier_complete_(bctx, cta);
                        lk.unlock();
                        bctx.cv.notify_all();
                    }
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    g_warpCtx        = nullptr;
    g_warpVoteCtx    = nullptr;
    g_warpVoteAllCtx = nullptr;
    g_blockSyncCtx   = nullptr;
    g_warpHistCtx    = nullptr;
    g_warpGroupCtx   = nullptr;
    g_warpTurnCtx    = nullptr;
}

// -----------------------------------------------------------------------------
// 2D Kernel launcher
//
// Interface:
//   struct dim3 { int x, y, z; };
//   void launchKernel2D(func, gridDim{x,y}, blockDim{x,y}, args...)
//
// Maps 2D grid/block to flat thread execution with proper SR_*_Y registers.
// -----------------------------------------------------------------------------

struct Dim3 { int x = 1, y = 1, z = 1; };

template<typename Func, typename... Args>
void launchKernel2D(Func func, Dim3 grid, Dim3 block, Args... args)
{
    off = ARG_BASE;
    (writeArg(args), ...);
    std::memcpy(g_constMemTemplate.data(), const_mem, sizeof(const_mem));

    const int threadsPerBlock = block.x * block.y * block.z;
    const int totalCTAs = grid.x * grid.y * grid.z;

    g_blockDim      = threadsPerBlock;
    g_gridDim       = totalCTAs;
    g_warpsPerBlock = (threadsPerBlock + WARP_SIZE - 1) / WARP_SIZE;
    const int totalWarps = g_warpsPerBlock * totalCTAs;

    std::vector<WarpShuffleCtx>  warpCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteCtx>     voteCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpVoteAllCtx>  voteAllCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpHmmaCtx>     hmmaCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpLdsmCtx>     ldsmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpStsmCtx>     stsmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpDmmaCtx>     dmmaCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpDmma884Ctx>  dmma884Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImma8816Ctx> imma8816Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImma16832Ctx> imma16832Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImmaSp16864Ctx> immaSp16864Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpImmaSpS4Ctx> immaSpS4Ctx(static_cast<size_t>(totalWarps));
    std::vector<WarpMovmCtx>     movmCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpHistCtx>     histCtx(static_cast<size_t>(totalWarps));
    std::vector<WarpGroupCtx>    groupCtx(static_cast<size_t>(totalWarps));
    g_warpCtx        = warpCtx.data();
    g_warpVoteCtx    = voteCtx.data();
    g_warpVoteAllCtx = voteAllCtx.data();
    g_warpHmmaCtx    = hmmaCtx.data();
    g_warpLdsmCtx    = ldsmCtx.data();
    g_warpStsmCtx    = stsmCtx.data();
    g_warpDmmaCtx    = dmmaCtx.data();
    g_warpDmma884Ctx = dmma884Ctx.data();
    g_warpImma8816Ctx = imma8816Ctx.data();
    g_warpImma16832Ctx = imma16832Ctx.data();
    g_warpImmaSp16864Ctx = immaSp16864Ctx.data();
    g_warpImmaSpS4Ctx = immaSpS4Ctx.data();
    g_warpMovmCtx    = movmCtx.data();
    g_warpHistCtx    = histCtx.data();
    g_warpGroupCtx   = groupCtx.data();

    std::vector<BlockSyncCtx> blockCtx(static_cast<size_t>(totalCTAs));
    g_blockSyncCtx = blockCtx.data();

    int ctaLinear = 0;
    for (int gz = 0; gz < grid.z; ++gz) {
    for (int gy = 0; gy < grid.y; ++gy) {
    for (int gx = 0; gx < grid.x; ++gx, ++ctaLinear) {

        std::memset(shared_mem, 0, SLIFTER_SHM_PAYLOAD_BYTES);  // payload only — tail is the PROT_NONE guard

        blockCtx[static_cast<size_t>(ctaLinear)].live = threadsPerBlock;
        for (int w = 0; w < g_warpsPerBlock; ++w) {
            const int wg = ctaLinear * g_warpsPerBlock + w;
            const int lanes = std::min(WARP_SIZE, threadsPerBlock - w * WARP_SIZE);
            warpCtx[static_cast<size_t>(wg)].live    = lanes;
            voteCtx[static_cast<size_t>(wg)].live    = lanes;
            voteAllCtx[static_cast<size_t>(wg)].live = lanes;
            hmmaCtx[static_cast<size_t>(wg)].live    = lanes;
            ldsmCtx[static_cast<size_t>(wg)].live    = lanes;
            stsmCtx[static_cast<size_t>(wg)].live    = lanes;
            dmmaCtx[static_cast<size_t>(wg)].live    = lanes;
            dmma884Ctx[static_cast<size_t>(wg)].live = lanes;
            imma8816Ctx[static_cast<size_t>(wg)].live = lanes;
            imma16832Ctx[static_cast<size_t>(wg)].live = lanes;
            immaSp16864Ctx[static_cast<size_t>(wg)].live = lanes;
            immaSpS4Ctx[static_cast<size_t>(wg)].live = lanes;
            movmCtx[static_cast<size_t>(wg)].live    = lanes;
            // Pre-publish lanes before spawn — see launchKernel.
            static const uint64_t initialRing2D[1] = {0xcbf29ce484222325ull};
            for (int l = 0; l < lanes; ++l) {
                histCtx[static_cast<size_t>(wg)].seq[l]  = 0;
                histCtx[static_cast<size_t>(wg)].hash[l] = 0xcbf29ce484222325ull;
                histCtx[static_cast<size_t>(wg)].ring[l] = initialRing2D;
            }
        }

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(threadsPerBlock));

        for (int tz = 0; tz < block.z; ++tz) {
        for (int ty = 0; ty < block.y; ++ty) {
        for (int tx = 0; tx < block.x; ++tx) {

            int flatTid = (tz * block.y + ty) * block.x + tx;
            threads.emplace_back([=]() {
                std::memcpy(const_mem, g_constMemTemplate.data(), sizeof(const_mem));
                std::memset(local_mem, 0, sizeof(local_mem));

                uint8_t* c0 = cmem0();
                // Block dimensions
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_X) = static_cast<uint32_t>(block.x);
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Y) = static_cast<uint32_t>(block.y);
                *reinterpret_cast<uint32_t*>(c0 + SR_NTID_Z) = static_cast<uint32_t>(block.z);
                // Grid dimensions
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_X) = static_cast<uint32_t>(grid.x);
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Y) = static_cast<uint32_t>(grid.y);
                *reinterpret_cast<uint32_t*>(c0 + SR_GRID_DIM_Z) = static_cast<uint32_t>(grid.z);
                // Block index
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_X) = static_cast<uint32_t>(gx);
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_Y) = static_cast<uint32_t>(gy);
                *reinterpret_cast<uint32_t*>(c0 + SR_CTAID_Z) = static_cast<uint32_t>(gz);
                // Thread index
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_X) = static_cast<uint32_t>(tx);
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_Y) = static_cast<uint32_t>(ty);
                *reinterpret_cast<uint32_t*>(c0 + SR_TID_Z) = static_cast<uint32_t>(tz);
                *reinterpret_cast<uint32_t*>(c0 + SR_LANE_ID) = static_cast<uint32_t>(flatTid & (WARP_SIZE - 1));

                g_threadCtx.cta         = ctaLinear;
                g_threadCtx.tid         = flatTid;
                g_threadCtx.lane        = flatTid & (WARP_SIZE - 1);
                g_threadCtx.warpInBlock = flatTid / WARP_SIZE;
                g_threadCtx.warpGlobal  = ctaLinear * g_warpsPerBlock + g_threadCtx.warpInBlock;

                g_laneHist = LaneHistory{};
                warp_hist_publish();

                func();

                // Lane exit — mirror launchKernel: leave group/collective
                // membership and barrier participation.
                {
                    const int wg2 = g_threadCtx.warpGlobal;
                    { std::lock_guard<std::mutex> lk(g_warpHistCtx[wg2].m);
                      g_warpHistCtx[wg2].ring[g_threadCtx.lane] = nullptr; }
                    g_warpGroupCtx[wg2].cv.notify_all();
                    { std::lock_guard<std::mutex> lk(g_warpCtx[wg2].m);        g_warpCtx[wg2].live--; }
                    g_warpCtx[wg2].cv.notify_all();
                    { std::lock_guard<std::mutex> lk(g_warpVoteCtx[wg2].m);    g_warpVoteCtx[wg2].live--; }
                    g_warpVoteCtx[wg2].cv.notify_all();
                    { std::lock_guard<std::mutex> lk(g_warpVoteAllCtx[wg2].m); g_warpVoteAllCtx[wg2].live--; }
                    g_warpVoteAllCtx[wg2].cv.notify_all();
                }
                {
                    BlockSyncCtx& bctx = g_blockSyncCtx[ctaLinear];
                    std::unique_lock<std::mutex> lk(bctx.m);
                    bctx.live--;
                    if (bctx.arrived >= bctx.live && bctx.arrived > 0) {
                        // Snapshot BAR.RED state too (see 1-D launch path).
                        block_barrier_complete_(bctx, ctaLinear);
                        lk.unlock();
                        bctx.cv.notify_all();
                    }
                }
            });

        }}}

        for (auto& t : threads) t.join();

    }}}

    g_warpCtx        = nullptr;
    g_warpVoteCtx    = nullptr;
    g_warpVoteAllCtx = nullptr;
    g_blockSyncCtx   = nullptr;
    g_warpHistCtx    = nullptr;
    g_warpGroupCtx   = nullptr;
}

// launchKernel3D — alias of launchKernel2D, which already loops grid.z/block.z
// and sets SR_*_Z. Hosts using a named 3D launch (e.g. GD_3d_cube) call this.
template<typename Func, typename... Args>
inline void launchKernel3D(Func func, Dim3 grid, Dim3 block, Args... args) {
    launchKernel2D(func, grid, block, args...);
}
