// Driver for the F2FP.TF32.F32.PACK_B fixture (PTX cvt.rn.tf32.f32).
//
// tf32 keeps f32's sign+exponent and the top 10 mantissa bits; the low 13
// mantissa bits are rounded away round-to-nearest-EVEN and zeroed. Result is a
// 32-bit bit pattern. A passthrough lift (no rounding) is value-wrong for any
// input whose low 13 mantissa bits are nonzero — which the global store here
// observes directly.
//
// Param offsets (sm90, base 0x210): 0x210 in, 0x218 out, 0x220 N.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_f2fp_tf32_packb();

// f32 -> tf32 (19-bit), round-to-nearest-even on the low 13 mantissa bits, the
// result kept as a full 32-bit bit pattern with the low 13 bits zeroed. This is
// the cheap uniform HW operation: round-bias the 32-bit pattern then mask off
// the low 13 bits. For finite normals this is exact RN-even. For Inf (low 13
// already 0) the bias+mask is a no-op -> Inf preserved. For an ordinary qNaN
// the mantissa high bit keeps it a NaN after masking. (A pathological NaN with
// nearly-all-ones mantissa is not exercised; tf32 NaN handling there is
// HW-defined and not value-relevant in the corpus, where every F2FP.TF32 result
// feeds an HMMA that re-masks the operands.)
static uint32_t f32_to_tf32_rn(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t lsb = (u >> 13) & 1;            // bit 13 = LSB of the kept field
    uint32_t rounded = u + 0x0FFFu + lsb;    // add half-1 + (lsb for tie-to-even)
    return rounded & 0xFFFFE000u;            // zero the low 13 bits
}

int main() {
    constexpr int N = 64;
    std::vector<float> in(N);
    std::vector<uint32_t> out(N, 0xDEADBEEF);

    // Exercise: low-13-bits set (forces rounding to matter), exact tf32 values
    // (low 13 already zero), RN-even ties (bit12 set, bits<12 zero), carry that
    // ripples into the exponent, negatives, subnormals, and specials.
    for (int i = 0; i < N; ++i) {
        float v;
        switch (i % 8) {
            case 0: { uint32_t b = 0x3F800000u | (((i + 1) * 0x111u) & 0x1FFFu); std::memcpy(&v, &b, 4); } break; // 1.0 + low-13 mantissa bits
            case 1: { uint32_t b = 0x40490FDBu; std::memcpy(&v, &b, 4); } break;  // pi (0x...0FDB, low bits set)
            case 2: { uint32_t b = 0x3F801000u; std::memcpy(&v, &b, 4); } break;  // tie: bit12 set exactly, kept-LSB even -> stays (round to even)
            case 3: { uint32_t b = 0x3F803000u; std::memcpy(&v, &b, 4); } break;  // tie: bit12 set, kept-LSB odd -> rounds up
            case 4:  v = -2.7182817f; break;                                       // negative, low bits set
            case 5: { uint32_t b = 0x4EFFFFFFu; std::memcpy(&v, &b, 4); } break;  // finite, all low mantissa set -> rounding carry ripples into exponent
            case 6: { uint32_t b = 0x00401234u; std::memcpy(&v, &b, 4); } break;  // tiny normal with low-13 bits set
            case 7:  v = (float)((i * 12345) % 99991) * 1.000123f; break;          // pseudo-random magnitude
        }
        in[i] = v;
    }

    launchKernel(kern_f2fp_tf32_packb, /*grid*/2, /*block*/32,
                 ArgAt{0x210}, in.data(),
                 ArgAt{0x218}, out.data(),
                 ArgAt{0x220}, N);

    int nbad = 0;
    for (int i = 0; i < N; ++i) {
        uint32_t fin; std::memcpy(&fin, &in[i], 4);
        uint32_t exp = f32_to_tf32_rn(in[i]);
        if (out[i] != exp) {
            if (nbad < 6)
                printf("MISMATCH %d: in=0x%08x got=0x%08x exp=0x%08x\n",
                       i, fin, out[i], exp);
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
