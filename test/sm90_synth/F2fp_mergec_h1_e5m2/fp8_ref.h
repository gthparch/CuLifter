// fp8_ref.h — portable (no-CUDA) software reference for f32 -> fp8 byte,
// RN-to-even, SATFINITE. Bit-exact with PTX cvt.rn.satfinite.{e4m3,e5m2}x2.f32
// (validated against cuda_fp8.hpp's software path). Used by the F2FP MERGE_C
// microbench oracles so they build under the standard test harness (which does
// not add a CUDA include path).
#pragma once
#include <cstdint>
#include <cstring>

// fmt: 0 = E4M3 (1.4.3 bias7, max 448, S.1111.111=NaN, no Inf)
//      1 = E5M2 (1.5.2 bias15, max 57344, Inf/NaN like f16>>8)
static inline uint8_t f32_to_fp8_ref(float f, int fmt) {
    uint32_t u; std::memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 1u;
    uint32_t au = u & 0x7FFFFFFFu;
    int fexp = (int)((au >> 23) & 0xFFu);
    uint32_t fman = au & 0x7FFFFFu;

    int MBITS, BIAS, MAXFIN, NANCODE, INFCODE, HAS_INF;
    if (fmt == 1) { MBITS = 2; BIAS = 15; MAXFIN = 0x7B; NANCODE = 0x7F; INFCODE = 0x7C; HAS_INF = 1; }
    else          { MBITS = 3; BIAS = 7;  MAXFIN = 0x7E; NANCODE = 0x7F; INFCODE = 0;    HAS_INF = 0; }

    uint8_t sign8 = (uint8_t)(sign << 7);

    // NaN / Inf in the source.
    int is_inf_nan = (fexp == 0xFF);
    int is_nan = (au > 0x7F800000u);

    uint32_t sig = fman | (1u << 23);
    int E = fexp - 127 + BIAS;
    int DROP = 23 - MBITS;
    int is_sub = (E <= 0);
    int extra = is_sub ? (1 - E) : 0;
    int shift = DROP + extra;
    if (shift > 31) shift = 31;

    uint32_t kept = sig >> shift;
    uint32_t shmask = (shift >= 32) ? 0xFFFFFFFFu : ((1u << shift) - 1u);
    uint32_t rem = sig & shmask;
    uint32_t halfbit = (shift >= 1) ? (1u << (shift - 1)) : 0u;
    int round_up = (rem > halfbit) || ((rem == halfbit) && (kept & 1u));
    uint32_t rounded = kept + (round_up ? 1u : 0u);

    int Em1 = is_sub ? 0 : (E - 1);
    uint32_t em = rounded + ((uint32_t)Em1 << MBITS);

    int overflow = (em > (uint32_t)MAXFIN);
    uint32_t spec;
    if (HAS_INF) spec = is_nan ? NANCODE : (is_inf_nan ? (uint32_t)INFCODE : (uint32_t)MAXFIN);
    else         spec = is_nan ? NANCODE : (uint32_t)MAXFIN;

    uint32_t mag = (is_inf_nan || overflow) ? spec : em;
    return (uint8_t)(mag | sign8);
}
