// Integer-division microbench: bisect the lifter bug for ptxas's runtime
// unsigned-division pattern (I2F.RP + MUFU.RCP + IADD3 magic + F2I + IMAD.HI
// + 3-step correction). Each kernel exercises one component or combination.
//
// All divisors come in as runtime kernel args so ptxas can't constant-fold
// to a shift; that forces emission of the reciprocal-multiplication scheme
// that the SLifter NVVM lifter currently mishandles. tensor.cu / tensor1.cu /
// tensor_microbench::tm_kloop_runtime all hit this.
#include <cstdint>

// ============================================================================
// 1. Pure division: per-thread t / d, where d is a runtime arg.
//    Simplest reproduction of the magic-mul pattern.
// ============================================================================
extern "C" __global__ void dm_div(int d, int *out) {
    int t = threadIdx.x;
    out[t] = t / d;
}

// ============================================================================
// 2. Pure modulo: per-thread t % d. Same SASS pattern (q is computed
//    internally and r = t - q*d), but exercises the correction step.
// ============================================================================
extern "C" __global__ void dm_mod(int d, int *out) {
    int t = threadIdx.x;
    out[t] = t % d;
}

// ============================================================================
// 3. Both div and mod from the same divisor. ptxas often shares the magic
//    constant; verifies that both consumers get the right value.
// ============================================================================
extern "C" __global__ void dm_divmod(int d, int *q_out, int *r_out) {
    int t = threadIdx.x;
    q_out[t] = t / d;
    r_out[t] = t % d;
}

// ============================================================================
// 4. Two independent divisors in one kernel. Forces two distinct magic
//    constants in the same SASS body.
// ============================================================================
extern "C" __global__ void dm_two_divisors(int d1, int d2, int *out1, int *out2) {
    int t = threadIdx.x;
    out1[t] = t / d1;
    out2[t] = t / d2;
}

// ============================================================================
// 5. warpId / warpN — mirrors tensor.cu's exact use site. Driven by
//    blockIdx + threadIdx and an arg-derived warpN. If only this fails
//    (and dm_div passes), the bug is in chaining the divide with a derived
//    dividend, not in the divide itself.
// ============================================================================
extern "C" __global__ void dm_warpid(int N, int *out) {
    const int warpN = (N + 16 - 1) / 16;
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if ((threadIdx.x & 31) == 0) {
        out[warpId * 2 + 0] = warpId / warpN;
        out[warpId * 2 + 1] = warpId % warpN;
    }
}

// ============================================================================
// 6. Division of a runtime dividend by a runtime divisor. Exercises the
//    pattern WITHOUT threadIdx — pure scalar.
// ============================================================================
extern "C" __global__ void dm_scalar(int dividend, int divisor, int *out) {
    if (threadIdx.x == 0) {
        out[0] = dividend / divisor;
        out[1] = dividend % divisor;
    }
}
