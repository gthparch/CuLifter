// Driver for the I2F.RP / I2F.U32.RP fixture (PTX cvt.rp.f32.{s32,u32}).
//
// .RP = round toward +Inf. For |int| >= 2^24 the int is not exactly f32-
// representable and .RP picks the next f32 >= x. A lifter that emits a plain
// sitofp/uitofp (round-to-nearest-even) gives a DIFFERENT f32 for those inputs.
//
// Oracle: convert with the hardware FP rounding mode set to FE_UPWARD, which is
// exactly round-toward-+Inf, matching cvt.rp.
//
// Param offsets (sm90): 0x210 in, 0x218 out_s, 0x220 out_u, 0x228 N.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_i2f_rp();

// Reliable round-toward-+Inf without depending on fesetround (which the
// compiler may ignore for int->float without FENV_ACCESS). long double holds
// every i32/u32 exactly; RN to float then nudge up by one ULP iff RN landed
// strictly below the true value.
static float rp_from_ld(long double ld) {
    float fn = (float)ld;
    if ((long double)fn < ld)
        fn = nextafterf(fn, INFINITY);
    return fn;
}
static float s32_to_f32_rp(int32_t v)  { return rp_from_ld((long double)v); }
static float u32_to_f32_rp(uint32_t v) { return rp_from_ld((long double)v); }

int main() {
    constexpr int N = 64;
    std::vector<int32_t> in(N);
    std::vector<float> out_s(N, -1.0f), out_u(N, -1.0f);

    // Cover small (exact) values AND large magnitudes (>2^24) where the rounding
    // direction is observable: values just above a power-of-two-stride float so
    // nearest rounds DOWN but .RP must round UP.
    for (int i = 0; i < N; ++i) {
        int32_t v;
        switch (i % 8) {
            case 0: v = i;                       break; // small exact
            case 1: v = -i;                      break; // small negative exact
            case 2: v = (1 << 24) + 1 + i;       break; // > 2^24, not f32-exact (RP rounds up)
            case 3: v = (1 << 24) + 3;           break; // odd offset, RP != RN
            case 4: v = 0x7FFFFFFF - i;          break; // near INT_MAX (huge, RP up)
            case 5: v = (int32_t)(0x80000001u + (uint32_t)i); break; // negative near INT_MIN as signed; as u32 = huge
            case 6: v = -(1 << 24) - 3;          break; // large negative, RP toward 0/up
            case 7: v = (int32_t)(0xFFFFFF81u);  break; // as u32 = 4294967169 (huge), as s32 = -127
        }
        in[i] = v;
    }

    launchKernel(kern_i2f_rp, /*grid*/2, /*block*/32,
                 ArgAt{0x210}, in.data(),
                 ArgAt{0x218}, out_s.data(),
                 ArgAt{0x220}, out_u.data(),
                 ArgAt{0x228}, N);

    int nbad = 0;
    for (int i = 0; i < N; ++i) {
        float es = s32_to_f32_rp(in[i]);
        float eu = u32_to_f32_rp((uint32_t)in[i]);
        uint32_t gsb, gub, esb, eub;
        std::memcpy(&gsb, &out_s[i], 4); std::memcpy(&esb, &es, 4);
        std::memcpy(&gub, &out_u[i], 4); std::memcpy(&eub, &eu, 4);
        if (gsb != esb || gub != eub) {
            if (nbad < 6)
                printf("MISMATCH %d (in=%d / u=%u): "
                       "s got=%.9g(0x%08x) exp=%.9g(0x%08x) | "
                       "u got=%.9g(0x%08x) exp=%.9g(0x%08x)\n",
                       i, in[i], (uint32_t)in[i],
                       out_s[i], gsb, es, esb,
                       out_u[i], gub, eu, eub);
            ++nbad;
        }
    }
    if (nbad) {
        printf("TEST FAILED (%d/%d)\n", nbad, N);
        return 1;
    }
    printf("TEST PASSED\n");
    return 0;
}
