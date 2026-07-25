// Driver for the I2F.S64.RP / I2F.U64.RP fixture (PTX cvt.rp.f32.{s64,u64}).
// 64-bit int -> f32, round toward +Inf. Oracle uses FE_UPWARD.
//
// Param offsets (sm90): 0x210 in, 0x218 out_s, 0x220 out_u, 0x228 N.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_i2f_rp64();

// Reliable round-toward-+Inf without depending on fesetround (which the
// compiler may ignore for int->float without FENV_ACCESS). long double (80-bit)
// holds every i64/u64 exactly; RN to float then nudge up by one ULP iff RN
// landed strictly below the true value.
static float rp_from_ld(long double ld) {
    float fn = (float)ld;                       // round-to-nearest
    if ((long double)fn < ld)                   // RN rounded DOWN -> bump up
        fn = nextafterf(fn, INFINITY);
    return fn;
}
static float s64_to_f32_rp(int64_t v)  { return rp_from_ld((long double)v); }
static float u64_to_f32_rp(uint64_t v) { return rp_from_ld((long double)v); }

int main() {
    constexpr int N = 64;
    std::vector<int64_t> in(N);
    std::vector<float> out_s(N, -1.0f), out_u(N, -1.0f);

    for (int i = 0; i < N; ++i) {
        int64_t v;
        switch (i % 8) {
            case 0: v = i;                                    break; // small exact
            case 1: v = -(int64_t)i - 1;                      break; // small negative (u64 = huge)
            case 2: v = (1LL << 40) + 1 + i;                  break; // huge, not f32-exact, RP up
            case 3: v = (1LL << 52) + 3;                      break; // big, RP != RN
            case 4: v = 0x7FFFFFFFFFFFFFFFLL - i;             break; // near INT64_MAX
            case 5: v = (int64_t)0x8000000000000001ULL + i;   break; // near INT64_MIN signed; u64 huge
            case 6: v = -((1LL << 40) + 5);                   break; // large negative
            case 7: v = (int64_t)0xFFFFFFFFFFFFFF81ULL;       break; // s64 = -127; u64 = ~2^64
        }
        in[i] = v;
    }

    launchKernel(kern_i2f_rp64, /*grid*/2, /*block*/32,
                 ArgAt{0x210}, in.data(),
                 ArgAt{0x218}, out_s.data(),
                 ArgAt{0x220}, out_u.data(),
                 ArgAt{0x228}, N);

    int nbad = 0;
    for (int i = 0; i < N; ++i) {
        float es = s64_to_f32_rp(in[i]);
        float eu = u64_to_f32_rp((uint64_t)in[i]);
        uint32_t gsb, gub, esb, eub;
        std::memcpy(&gsb, &out_s[i], 4); std::memcpy(&esb, &es, 4);
        std::memcpy(&gub, &out_u[i], 4); std::memcpy(&eub, &eu, 4);
        if (gsb != esb || gub != eub) {
            if (nbad < 6)
                printf("MISMATCH %d (in=%lld / u=%llu): "
                       "s got=0x%08x exp=0x%08x | u got=0x%08x exp=0x%08x\n",
                       i, (long long)in[i], (unsigned long long)(uint64_t)in[i],
                       gsb, esb, gub, eub);
            ++nbad;
        }
    }
    if (nbad) { printf("TEST FAILED (%d/%d)\n", nbad, N); return 1; }
    printf("TEST PASSED\n");
    return 0;
}
