// Microbench: the sm90 mbarrier "arrive-on-cp.async" op
//   ARRIVES.LDGSTSBAR.64.TRANSCNT [smem_mbar]
// emitted by `cp.async.mbarrier.arrive.shared::cta.b64 [mbar]` (the
// FlashAttention-forward producer→consumer handshake that increments the
// mbarrier's LDGSTS transaction count). Paired with a cp.async (LDGSTS) fill
// of shared memory and a try_wait consumer spin — the exact protocol shape
// flash_fwd_hdimdiff uses (243× ARRIVES + SYNCS.PHASECHK.TRYWAIT waits).
//
// Exercises the lifter's ARRIVES handler. On x86 (synchronous, single-lane)
// the paired SYNCS.PHASECHK.TRYWAIT wait already resolves always-ready
// (lift_syncs), so the arrive that increments the count it would check is a
// NOP — analogous to the SM90 x86 NOP set (SYNCS.ARRIVE / MEMBAR / DEPBAR).
//
// Oracle (host_cpu.cpp): cp.async copies in[i] -> smem[i], so out[i]==in[i]+1.
#include <cstdint>
#include <cuda.h>

extern "C" __global__ void mbarrier_arrive(const float* __restrict__ in,
                                           float* out, int N) {
    __shared__ float smem[256];
    __shared__ __align__(8) uint64_t mbar;
    int tid = threadIdx.x;

    if (tid == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "l"(&mbar), "r"(blockDim.x));
    }
    __syncthreads();

    // cp.async (LDGSTS) global -> shared
    uint32_t smem_int = (uint32_t)__cvta_generic_to_shared(&smem[tid]);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n\t"
                 :: "r"(smem_int), "l"(&in[tid]));

    // mbarrier arrive that tracks the cp.async transaction count
    //   -> ARRIVES.LDGSTSBAR.64.TRANSCNT [mbar]
    uint32_t mbar_int = (uint32_t)__cvta_generic_to_shared(&mbar);
    asm volatile("cp.async.mbarrier.arrive.shared::cta.b64 [%0];"
                 :: "r"(mbar_int));

    // consumer wait (try_wait spin) -> SYNCS.PHASECHK.TRANS64.TRYWAIT
    uint32_t phase = 0;
    asm volatile(
        "{\n\t .reg .pred P;\n\t L1:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra L1;\n\t }\n\t"
        :: "r"(mbar_int), "r"(phase));

    if (tid < N) out[tid] = smem[tid] + 1.0f;
}
