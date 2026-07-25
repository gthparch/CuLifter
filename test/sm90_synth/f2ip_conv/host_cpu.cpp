#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"

// Driver for the F2IP.S8.F32 fixture. The kernel quantizes 4 fp32 values into a
// packed int8x4 word (the CUTLASS NumericArrayConverter<int8_t, float, 4> epilogue):
//   out[i] (as 4 bytes) = { s8(in[i*4+0]), s8(in[i*4+1]), s8(in[i*4+2]), s8(in[i*4+3]) }
// where s8(x) = saturate-to-[-128,127]( round-to-nearest-even(x) ).
//
// Param offsets (sm90, base 0x210): 0x210 in, 0x218 out, 0x220 N.

extern "C" void kern_f2ip();

// Round-to-nearest-even then signed-saturate to int8 — the F2IP.S8.F32.NTZ oracle.
static int8_t f2ip_s8(float x) {
    float r = std::nearbyint(x);          // round-to-nearest-even (default FE mode)
    if (r >  127.0f) return  127;
    if (r < -128.0f) return -128;
    return static_cast<int8_t>(r);
}

int main() {
    constexpr int N = 64;                 // 64 packed words = 256 floats
    std::vector<float> in(N * 4);
    std::vector<int32_t> out(N, 0x5A5A5A5A);

    // Mix of values: exact ints, .5 ties (test round-to-even), and out-of-range
    // magnitudes (test saturation to [-128,127]).
    for (int i = 0; i < N * 4; ++i) {
        switch (i % 8) {
            case 0: in[i] =  static_cast<float>((i % 200) - 100);     break; // in-range int
            case 1: in[i] =  2.5f;   break;   // tie -> 2 (round-to-even)
            case 2: in[i] =  3.5f;   break;   // tie -> 4 (round-to-even)
            case 3: in[i] = -2.5f;   break;   // tie -> -2
            case 4: in[i] =  500.0f; break;   // saturate -> 127
            case 5: in[i] = -500.0f; break;   // saturate -> -128
            case 6: in[i] =  1.4999f; break;  // -> 1
            case 7: in[i] = -0.5f;   break;   // tie -> 0 (round-to-even)
        }
    }

    launchKernel(kern_f2ip, /*grid*/2, /*block*/32,
                 ArgAt{0x210}, in.data(),
                 ArgAt{0x218}, out.data(),
                 ArgAt{0x220}, N);

    int nbad = 0;
    for (int i = 0; i < N; ++i) {
        uint8_t exp_b[4];
        for (int j = 0; j < 4; ++j)
            exp_b[j] = static_cast<uint8_t>(f2ip_s8(in[i * 4 + j]));
        uint32_t exp;
        std::memcpy(&exp, exp_b, 4);
        uint32_t got = static_cast<uint32_t>(out[i]);
        if (got != exp) {
            if (!nbad)
                printf("first bad i=%d got=0x%08X exp=0x%08X "
                       "(in=%g,%g,%g,%g)\n",
                       i, got, exp,
                       in[i*4+0], in[i*4+1], in[i*4+2], in[i*4+3]);
            nbad++;
        }
    }
    if (!nbad) {
        printf("REF VERIFIED (%d packed words)\nTEST PASSED\n", N);
        return 0;
    }
    printf("FAILED: %d/%d mismatch\n", nbad, N);
    return 1;
}
