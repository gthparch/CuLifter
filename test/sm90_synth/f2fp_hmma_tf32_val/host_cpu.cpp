// host_cpu.cpp — VALUE driver for f2fp_hmma_tf32_val (F2FP.TF32 -> HMMA composition).
//
// Feeds RAW f32 A/B inputs whose mantissa bit12 (0x1000) is set, so the HW RN-even
// tf32 conversion (the fix: u + 0x0FFF + ((u>>13)&1) ; & 0xFFFFE000) and plain
// TRUNCATION (& 0xFFFFE000) DIFFER by 1 tf32-ULP. The oracle uses RN-even and the
// comparison is BIT-EXACT, so a truncating lift FAILS and only the RN-even fix passes.
// The driver ALSO computes a truncation oracle and asserts the lifted result MATCHES
// rneven and DIVERGES from trunc -> the test is a genuine guard for the 20th fix, not
// a tolerance pass.
//
// Accumulation order mirrors the x86 hmma1688_tf32_warp wrapper: acc=C[m][n];
// for k: acc += Atf[m][k]*Btf[k][n]  (sequential f32).
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"
extern "C" void kern_f2fp_hmma_tf32_val();

static uint32_t bits(float f){ uint32_t u; std::memcpy(&u, &f, 4); return u; }
static float    fromu(uint32_t u){ float f; std::memcpy(&f, &u, 4); return f; }

// HW RN-even f32 -> tf32 (the corpus cvt.rn.tf32.f32 == F2FP.TF32.F32.PACK_B).
static float tf32_rneven(float f){
    uint64_t u = bits(f);
    uint64_t lsb = (u >> 13) & 1ull;
    uint32_t r = (uint32_t)((u + 0x0FFFull + lsb) & 0xFFFFE000ull);
    return fromu(r);
}
// Plain truncation (the PRE-FIX effective behavior) — used only to PROVE divergence.
static float tf32_trunc(float f){ return fromu(bits(f) & 0xFFFFE000u); }

int main(){
    constexpr int M=16, N=8, K=8;
    std::vector<float> A(M*K), B(K*N), C(M*N), D(M*N, -7.0f);
    uint32_t s = 0xF2FA33u;
    auto nxt = [&](){ s = s*1664525u + 1013904223u; return s; };
    auto rnd = [&](float sc){ return sc*(((int)(nxt()>>16 & 0x3FF)-512)/256.0f); };
    // Inputs: random f32 in ~[-1,1], with the LOW 13 MANTISSA BITS forced to a NON-TIE
    // value strictly in (0x1000, 0x2000) — i.e. bit12 set AND lower bits set. There the
    // HW RN-even ROUNDS UP (away) while plain TRUNCATION rounds DOWN, so the two tf32
    // results differ by 1 ULP on EVERY operand -> a truncating lift produces a
    // bit-different GEMM output and FAILS the bit-exact check. (Forcing exactly 0x1000
    // would be an EXACT TIE where RN-to-even often matches truncation — useless here.)
    auto mk = [&](float sc){
        uint32_t u = bits(rnd(sc)) & 0xFFFFE000u;          // clear low 13
        uint32_t low = 0x1001u + (nxt() % 0x0FFEu);        // in (0x1000, 0x2000), non-tie
        return fromu(u | low);
    };
    for (int i=0;i<M*K;++i) A[i] = mk(1.0f);
    for (int i=0;i<K*N;++i) B[i] = mk(1.0f);
    for (int i=0;i<M*N;++i) C[i] = rnd(2.0f);   // C added at f32, no tf32 round

    launchKernel(kern_f2fp_hmma_tf32_val, 1, 32,
        ArgAt{0x210}, A.data(), ArgAt{0x218}, B.data(),
        ArgAt{0x220}, C.data(), ArgAt{0x228}, D.data());

    // Oracles (f32, wrapper accumulation order): rneven (expected) + trunc (must diverge).
    auto gemm = [&](float(*cvt)(float), std::vector<float>& out){
        out.assign(M*N, 0.0f);
        for (int m=0;m<M;++m) for (int n=0;n<N;++n){
            float acc = C[m*N+n];
            for (int k=0;k<K;++k) acc += cvt(A[m*K+k]) * cvt(B[k*N+n]);
            out[m*N+n] = acc;
        }
    };
    std::vector<float> Drn, Dtr;
    gemm(tf32_rneven, Drn);
    gemm(tf32_trunc,  Dtr);

    int bad_rn=0, bad_tr=0, ndiff_rn_tr=0;
    double maxabs_rn=0;
    for (int i=0;i<M*N;++i){
        if (bits(D[i]) != bits(Drn[i])) ++bad_rn;        // BIT-EXACT vs RN-even
        if (bits(D[i]) != bits(Dtr[i])) ++bad_tr;        // should DIFFER from trunc
        if (bits(Drn[i]) != bits(Dtr[i])) ++ndiff_rn_tr; // oracle-internal discriminator
        maxabs_rn = std::max(maxabs_rn, (double)std::fabs(D[i]-Drn[i]));
    }
    printf("F2FP.TF32->HMMA composition value check (M%d N%d K%d, bit12 forced on all operands):\n", M,N,K);
    printf("  lifted vs RN-even : bit-exact=%d/%d (bad=%d) maxabs=%.3e\n", M*N-bad_rn, M*N, bad_rn, maxabs_rn);
    printf("  lifted vs trunc   : bad=%d/%d  (expect LARGE: lifted must DIVERGE from truncation)\n", bad_tr, M*N);
    printf("  oracle rneven vs trunc differ on %d/%d cells (discriminator)\n", ndiff_rn_tr, M*N);

    // Guard: discriminator must be non-trivial (else the test wouldn't distinguish the fix).
    if (ndiff_rn_tr < M*N/4) {
        printf("WEAK DISCRIMINATOR (%d/%d) — inputs did not exercise enough ties\nTEST FAILED\n", ndiff_rn_tr, M*N);
        return 1;
    }
    if (bad_rn == 0 && bad_tr > 0) {
        printf("REF VERIFIED — lifted matches RN-even tf32 BIT-EXACT and diverges from truncation "
               "(F2FP.TF32 RN-even fix confirmed through the HMMA composition)\nTEST PASSED\n");
        return 0;
    }
    printf("FAILED: bad_rn=%d (must be 0)  bad_tr=%d (must be >0)\nTEST FAILED\n", bad_rn, bad_tr);
    return 1;
}
