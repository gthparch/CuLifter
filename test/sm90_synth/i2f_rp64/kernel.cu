// I2F.S64.RP / I2F.U64.RP fixture — 64-bit int -> f32, round toward +Inf.
//
// Exercises the 64-bit-source directed-rounding path of lift_i2f. The corpus
// has I2F.U64.RP (104 sites) and I2F.S64 (RN). PTX cvt.rp.f32.s64 / .u64 lower
// to I2F.S64.RP / I2F.U64.RP. For |int| >= 2^24 the f32 cannot hold the value
// exactly and .RP must pick the next f32 >= x — the round-trip correction in
// the lift must use a wide enough int (>64 bits) so a rounded-up result does
// not overflow the comparison.
#include <cstdint>

extern "C" __global__ void kern_i2f_rp64(const int64_t* __restrict__ in,
                                         float* __restrict__ out_s,
                                         float* __restrict__ out_u,
                                         int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int64_t s = in[i];
    uint64_t u = (uint64_t)s;
    float rs, ru;
    asm volatile("cvt.rp.f32.s64 %0, %1;" : "=f"(rs) : "l"(s));   // -> I2F.S64.RP
    asm volatile("cvt.rp.f32.u64 %0, %1;" : "=f"(ru) : "l"(u));   // -> I2F.U64.RP
    out_s[i] = rs;
    out_u[i] = ru;
}
