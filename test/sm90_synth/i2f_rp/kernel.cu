// I2F.RP / I2F.U32.RP fixture — int -> f32 with round-toward-POSITIVE-infinity.
//
// CUTLASS emits I2F.RP (1267 sites) and I2F.U32.RP (588 sites) on sm_90 — the
// NumericConverter / fast int->float paths that round up. PTX `cvt.rp.f32.s32`
// and `cvt.rp.f32.u32` lower to a single I2F.RP / I2F.U32.RP.
//
// .RP = round toward +Inf (ceil in float space). For |int| < 2^24 the convert
// is exact and .RP == round-to-nearest; the difference only appears when the
// integer is NOT exactly representable in f32 (|x| >= 2^24), where .RP must pick
// the next f32 >= x while the default sitofp/uitofp rounds to nearest-even.
//
// The kernel converts each int with cvt.rp twice (signed and unsigned view) and
// stores both f32 results, so the host oracle (which rounds up explicitly) can
// catch a lifter that emits a plain round-to-nearest sitofp/uitofp.
#include <cstdint>

extern "C" __global__ void kern_i2f_rp(const int32_t* __restrict__ in,
                                       float* __restrict__ out_s,
                                       float* __restrict__ out_u,
                                       int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int32_t s = in[i];
    uint32_t u = (uint32_t)s;
    float rs, ru;
    asm volatile("cvt.rp.f32.s32 %0, %1;" : "=f"(rs) : "r"(s));    // -> I2F.RP
    asm volatile("cvt.rp.f32.u32 %0, %1;" : "=f"(ru) : "r"(u));    // -> I2F.U32.RP
    out_s[i] = rs;
    out_u[i] = ru;
}
