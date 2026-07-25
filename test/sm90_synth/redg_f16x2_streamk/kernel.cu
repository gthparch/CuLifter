// redg_f16x2_streamk — emits REDG.E.ADD.F16x2.RN.STRONG.GPU (the cutlass stream-K
// split-K epilogue reduction opcode), via inline PTX red.global (fire-and-forget,
// no result reg) so ptxas picks REDG (reduce-only), not ATOM (return form).
//
// Two kernels:
//   k_redg_h2_unique  : each global thread reduces its packed half pair into a
//                       UNIQUE output slot -> SINGLE contributor per address ->
//                       deterministic / bit-exact (order-independent).
//   k_redg_h2_contend : threads reduce into a small bin set (idx[t]) -> MANY
//                       contributors per address -> order-tolerant (f64 oracle).
#include <cuda_fp16.h>

__device__ __forceinline__ void redg_add_h2(__half2* p, __half2 v) {
    unsigned bits = *reinterpret_cast<const unsigned*>(&v);
    asm volatile("red.global.add.noftz.f16x2 [%0], %1;" :: "l"(p), "r"(bits) : "memory");
}

extern "C" __global__ void k_redg_h2_unique(
        __half2* __restrict__ out, const __half2* __restrict__ a, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    redg_add_h2(out + t, a[t]);
}

extern "C" __global__ void k_redg_h2_contend(
        __half2* __restrict__ out, const __half2* __restrict__ a,
        const int* __restrict__ idx, int N) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;
    redg_add_h2(out + idx[t], a[t]);
}
