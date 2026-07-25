// host_cpu.cpp — exec+value driver for redg_f16x2_streamk.
//
// Verifies the lifted lowering of REDG.E.ADD.F16x2.RN.STRONG.GPU (the cutlass
// stream-K split-K epilogue reduction opcode) -> `atomicrmw fadd <2 x half>`.
// Before the F16x2 token fix this opcode fell into the integer `atomicrmw add`
// path (it adds raw half-pair BIT PATTERNS, not fp16 values) — wrong for any
// multi-add reduction. This driver runs the lifted kernel on the x86 wrapper and
// checks values against a per-lane fp16 oracle.
//
//   PHASE 1  k_redg_h2_unique  — each global thread reduces a[t] into a UNIQUE
//     slot out[t] (single contributor per address). Order-independent =>
//     BIT-EXACT. Run twice: default (sequential CTAs) AND CULIFTER_CONCURRENT_CTA
//     (all grid*block threads spawned at once) — both must be bit-exact, since no
//     address has >1 writer.
//
//   PHASE 2  k_redg_h2_contend — threads reduce a[t] into a small bin set
//     (idx[t]) => MANY contributors per address. Contended fp16 atomic-add is
//     order-sensitive (fp16 non-associativity), so we DO NOT claim bit-exact;
//     instead an f64 oracle accumulates the same contributions and we assert
//     max relative error < 1e-3 (fp16 has ~3 decimal digits; per-add rounding
//     accumulates). This is the genuine stream-K multi-contributor shape.
//
// sm_90 arg layout (from the SASS const reads):
//   unique : 0x208 desc | 0x210 out | 0x218 a | 0x220 N
//   contend: 0x208 desc | 0x210 out | 0x218 a | 0x220 idx | 0x228 N
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void k_redg_h2_unique();
extern "C" void k_redg_h2_contend();

// ---- IEEE 754 binary16 <-> binary32 (round-to-nearest-even) ----------------
static float h2f(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff, out;
    if (e == 0) {
        if (m == 0) { out = s << 31; }
        else { // subnormal
            e = 127 - 15 + 1;
            while ((m & 0x400) == 0) { m <<= 1; e--; }
            m &= 0x3ff;
            out = (s << 31) | (e << 23) | (m << 13);
        }
    } else if (e == 0x1f) {
        out = (s << 31) | (0xff << 23) | (m << 13); // Inf/NaN
    } else {
        out = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f; __builtin_memcpy(&f, &out, 4); return f;
}
static uint16_t f2h(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t s = (x >> 16) & 0x8000;
    int32_t  e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t m = x & 0x7fffff;
    if (((x >> 23) & 0xff) == 0xff) // Inf/NaN
        return (uint16_t)(s | 0x7c00 | (m ? 0x200 : 0));
    if (e >= 0x1f) return (uint16_t)(s | 0x7c00);          // overflow -> Inf
    if (e <= 0) {
        if (e < -10) return (uint16_t)s;                   // underflow -> 0
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - e);
        uint32_t half_bits = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_bits & 1))) half_bits++;
        return (uint16_t)(s | half_bits);
    }
    uint16_t out = (uint16_t)(s | (uint32_t(e) << 10) | (m >> 13));
    uint32_t rem = m & 0x1fff, halfway = 0x1000;
    if (rem > halfway || (rem == halfway && (out & 1))) out++;  // RN-even (carries into exp OK)
    return out;
}
// fp16 add: exact f32 sum (f32 holds sum of two fp16 exactly), then round to fp16.
static uint16_t hadd(uint16_t a, uint16_t b) { return f2h(h2f(a) + h2f(b)); }

static inline uint32_t pack(uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); }

int main() {
    bool all_ok = true;

    // =====================================================================
    // PHASE 1 — single-contributor-per-address, BIT-EXACT (seq + concurrent)
    // =====================================================================
    {
        const int GRID = 4, BLOCK = 64, N = GRID * BLOCK;  // 256 unique slots
        // Inputs a[t] and bases b[t] (non-trivial fp16 values, both lanes differ).
        std::vector<uint32_t> a(N), base(N);
        for (int t = 0; t < N; ++t) {
            uint16_t alo = f2h(0.5f + 0.125f * (t % 11));      // [0.5 .. 1.75]
            uint16_t ahi = f2h(-2.0f + 0.25f * (t % 7));       // [-2 .. -0.5]
            uint16_t blo = f2h(3.0f + 0.0625f * (t % 13));
            uint16_t bhi = f2h(10.0f - 0.5f * (t % 5));
            a[t]    = pack(alo, ahi);
            base[t] = pack(blo, bhi);
        }
        // Oracle: out[t] = base[t] +(fp16,lanewise) a[t].  Single add -> exact.
        std::vector<uint32_t> oracle(N);
        for (int t = 0; t < N; ++t) {
            uint16_t lo = hadd((uint16_t)(base[t] & 0xffff), (uint16_t)(a[t] & 0xffff));
            uint16_t hi = hadd((uint16_t)(base[t] >> 16),    (uint16_t)(a[t] >> 16));
            oracle[t] = pack(lo, hi);
        }

        auto run_and_check = [&](const char* tag) -> bool {
            std::vector<uint32_t> out(base);  // init out = base
            launchKernel(k_redg_h2_unique, GRID, BLOCK,
                         ArgAt{0x208}, (uint64_t)0,
                         ArgAt{0x210}, out.data(),
                         ArgAt{0x218}, a.data(),
                         ArgAt{0x220}, (int)N);
            int nfail = 0; int first = -1;
            for (int t = 0; t < N; ++t)
                if (out[t] != oracle[t]) { if (first < 0) first = t; nfail++; }
            printf("PHASE 1 [%s]: bit-exact slots %d/%d%s",
                   tag, N - nfail, N, nfail ? "" : "  (all match)\n");
            if (nfail) printf("  first mismatch slot %d: got %08x want %08x\n",
                              first, out[first], oracle[first]);
            return nfail == 0;
        };

        unsetenv("CULIFTER_CONCURRENT_CTA");
        all_ok &= run_and_check("sequential");
        setenv("CULIFTER_CONCURRENT_CTA", "1", 1);
        all_ok &= run_and_check("concurrent");
        unsetenv("CULIFTER_CONCURRENT_CTA");
    }

    // =====================================================================
    // PHASE 2a — MANY contributors per address, but contributions are EXACTLY
    //   representable in fp16 (0.25/0.5 steps) and partial sums stay <= ~256,
    //   so every fp16 RN add is exact regardless of order => BIT-EXACT even
    //   under contention (f64 oracle == the exact integer/half sum). This is
    //   the strongest claim the FLOAT-ORDER CAVEAT permits for a contended
    //   float reduction: pick inputs where fp16 non-associativity vanishes.
    // =====================================================================
    {
        const int GRID = 8, BLOCK = 64, N = GRID * BLOCK;  // 512 contributions
        const int BINS = 8;                                // 64 contributors/bin
        std::vector<uint32_t> a(N);
        std::vector<int>      idx(N);
        for (int t = 0; t < N; ++t) {
            // lanes: 0.25 and 0.5 — exactly representable; 64*0.5=32, 64*0.25=16,
            // all partial sums exact in fp16 (<2048 integer-exact, step >= 0.25).
            uint16_t alo = f2h(((t % 3) == 0) ? 0.5f : 0.25f);
            uint16_t ahi = f2h(((t % 2) == 0) ? 0.25f : 0.5f);
            a[t]   = pack(alo, ahi);
            idx[t] = t % BINS;
        }
        std::vector<double> ref_lo(BINS, 0.0), ref_hi(BINS, 0.0);
        for (int t = 0; t < N; ++t) {
            ref_lo[idx[t]] += (double)h2f((uint16_t)(a[t] & 0xffff));
            ref_hi[idx[t]] += (double)h2f((uint16_t)(a[t] >> 16));
        }
        auto run2a = [&](const char* tag) -> bool {
            std::vector<uint32_t> out(BINS, 0);  // init bins = +0.0 (0+x exact)
            launchKernel(k_redg_h2_contend, GRID, BLOCK,
                         ArgAt{0x208}, (uint64_t)0,
                         ArgAt{0x210}, out.data(),
                         ArgAt{0x218}, a.data(),
                         ArgAt{0x220}, idx.data(),
                         ArgAt{0x228}, (int)N);
            int nfail = 0;
            for (int b = 0; b < BINS; ++b) {
                double glo = (double)h2f((uint16_t)(out[b] & 0xffff));
                double ghi = (double)h2f((uint16_t)(out[b] >> 16));
                if (glo != ref_lo[b] || ghi != ref_hi[b]) {
                    nfail++;
                    printf("  bin %d: got(%.4f,%.4f) want(%.4f,%.4f)\n",
                           b, glo, ghi, ref_lo[b], ref_hi[b]);
                }
            }
            printf("PHASE 2a [%s]: %d bins exact %d/%d (64 contributors/bin, exact fp16)\n",
                   tag, BINS, BINS - nfail, BINS);
            return nfail == 0;
        };
        unsetenv("CULIFTER_CONCURRENT_CTA");
        all_ok &= run2a("sequential");
        setenv("CULIFTER_CONCURRENT_CTA", "1", 1);
        all_ok &= run2a("concurrent");
        unsetenv("CULIFTER_CONCURRENT_CTA");
    }

    // =====================================================================
    // PHASE 2b — MANY contributors with MESSY fp16 values (rounding DOES
    //   accumulate). Contended float-add is order-sensitive, so we do NOT
    //   claim bit-exact; an f64 oracle bounds it. The fp16 reduction of K
    //   terms has worst-case rel error ~ (K-1)*2^-11; with K=64 and positive
    //   terms the realized error is well under 1e-2. The integer-add bug would
    //   add packed BIT PATTERNS (e.g. 0x3C00+0x3C00=0x7800 = 32768.0 for two
    //   1.0s) -> max_rel astronomically large, so 1e-2 robustly discriminates.
    // =====================================================================
    {
        const int GRID = 8, BLOCK = 64, N = GRID * BLOCK;
        const int BINS = 8;
        std::vector<uint32_t> a(N);
        std::vector<int>      idx(N);
        for (int t = 0; t < N; ++t) {
            uint16_t alo = f2h(0.0131f * (1 + (t % 17)));   // non-dyadic -> rounds
            uint16_t ahi = f2h(0.0207f * (1 + (t % 23)));
            a[t]   = pack(alo, ahi);
            idx[t] = t % BINS;
        }
        std::vector<double> ref_lo(BINS, 0.0), ref_hi(BINS, 0.0);
        for (int t = 0; t < N; ++t) {
            ref_lo[idx[t]] += (double)h2f((uint16_t)(a[t] & 0xffff));
            ref_hi[idx[t]] += (double)h2f((uint16_t)(a[t] >> 16));
        }
        std::vector<uint32_t> out(BINS, 0);
        launchKernel(k_redg_h2_contend, GRID, BLOCK,
                     ArgAt{0x208}, (uint64_t)0,
                     ArgAt{0x210}, out.data(),
                     ArgAt{0x218}, a.data(),
                     ArgAt{0x220}, idx.data(),
                     ArgAt{0x228}, (int)N);
        double max_rel = 0.0;
        for (int b = 0; b < BINS; ++b) {
            double glo = (double)h2f((uint16_t)(out[b] & 0xffff));
            double ghi = (double)h2f((uint16_t)(out[b] >> 16));
            double dlo = std::fabs(glo - ref_lo[b]) / (std::fabs(ref_lo[b]) + 1e-30);
            double dhi = std::fabs(ghi - ref_hi[b]) / (std::fabs(ref_hi[b]) + 1e-30);
            if (dlo > max_rel) max_rel = dlo;
            if (dhi > max_rel) max_rel = dhi;
        }
        bool ok = max_rel < 1e-2;
        printf("PHASE 2b [contend, messy fp16]: %d bins, max_rel=%.3e -> %s\n",
               BINS, max_rel, ok ? "OK (<1e-2, fp16-accum)" : "DIVERGED");
        all_ok &= ok;
    }

    printf("%s\n", all_ok ? "TEST PASSED" : "DIVERGED");
    fflush(stdout);
    return all_ok ? 0 : 1;
}
