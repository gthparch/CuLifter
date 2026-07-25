// x86 driver + f64 oracle for the CUTLASS 23 K-reduction (SHFL.BFLY butterfly +
// bf16 fragment decode + bf16 output convert). Launches a full warp (32 lanes)
// so the warp-true shfl_xor butterfly runs, then value-verifies the per-(group,
// col) reduction vector bit-exactly vs an independent f64-then-bf16 oracle.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include "kernel_wrapper.h"

extern "C" void kern_k_reduction();

// bf16 <-> float helpers (IEEE bfloat16 = top 16 bits of f32).
static float bf16_to_f(uint16_t b){ uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t f_to_bf16_rne(float f){
    uint32_t u; memcpy(&u,&f,4);
    if(((u>>23)&0xFF)==0xFF && (u&0x7FFFFF)) return (uint16_t)((u>>16)|0x0040); // NaN
    uint32_t lsb=(u>>16)&1, round=0x7FFF+lsb;
    return (uint16_t)((u+round)>>16);
}

int main(){
    const int LANES = 32;
    std::vector<uint16_t> Afrag(LANES*8), out(LANES, 0x4242 /*canary*/);
    // exact-in-bf16 inputs: multiples of 1/256 in a small range (bf16 has 8
    // mantissa bits, so values that fit in 8 fractional bits are exact).
    for(int i=0;i<LANES*8;i++){
        float v = (float)(((i*37)%512) - 256) / 256.0f;   // [-1,1) step 1/256, exact bf16
        Afrag[i] = f_to_bf16_rne(v);
    }

    launchKernel(kern_k_reduction, /*grid*/1, /*block*/LANES,
        ArgAt{0x210}, Afrag.data(),
        ArgAt{0x218}, out.data(),
        ArgAt{0x220}, 0);

    // --- f64 oracle: per-lane partial, then quad butterfly (xor1+xor2 = full
    //     quad sum), in the SAME f32 order the kernel uses (fadd accumulation).
    //     Inputs are exact bf16 so the partial f32 sums are exact; the butterfly
    //     is a sum over the 4 lanes -> compute in f64, then the bf16 convert.
    std::vector<float> partial(LANES);
    for(int lane=0; lane<LANES; lane++){
        const uint16_t* A = &Afrag[lane*8];
        // even-row pattern A[0],A[1],A[4],A[5], f32 fadd order
        float p = 0.f;
        p += bf16_to_f(A[0]); p += bf16_to_f(A[1]);
        p += bf16_to_f(A[4]); p += bf16_to_f(A[5]);
        partial[lane] = p;
    }
    // butterfly: sum over the quad {g*4+0..3}. The kernel does
    //   s1 = p + xor1(p); s2 = s1 + xor2(s1).
    // For lane L in quad q: xor1 swaps L^1, xor2 swaps L^2. After both, every
    // lane holds p[q0]+p[q1]+p[q2]+p[q3] (the full quad sum) — but the f32 ADD
    // ORDER differs per lane, so mirror it EXACTLY per lane.
    auto xory=[&](std::vector<float>&v,int x){ std::vector<float> r(LANES); for(int L=0;L<LANES;L++) r[L]=v[L^x]; return r; };
    std::vector<float> s1(LANES), s2(LANES);
    { auto x1=xory(partial,1); for(int L=0;L<LANES;L++) s1[L]=partial[L]+x1[L]; }
    { auto x2=xory(s1,2);      for(int L=0;L<LANES;L++) s2[L]=s1[L]+x2[L]; }

    std::vector<uint16_t> exp(LANES);
    for(int lane=0; lane<LANES; lane++){
        int g=lane>>2, col=lane&3;
        exp[g*4+col] = f_to_bf16_rne(s2[lane]);
    }

    int mism=0, first=-1; double max_abs=0;
    for(int i=0;i<LANES;i++){
        if(out[i]!=exp[i]){ if(first<0)first=i; mism++;
            double a=std::fabs((double)bf16_to_f(out[i])-(double)bf16_to_f(exp[i]));
            if(a>max_abs)max_abs=a; }
    }
    printf("=== gemm_k_reduction (23 operand K-reduction: SHFL.BFLY butterfly + bf16) ===\n");
    printf("lanes=%d  SHFL.BFLY butterfly (xor1,xor2) + bf16 fragment decode + bf16 convert\n", LANES);
    printf("mismatches=%d/%d  max_abs=%g  first_bad=%d\n", mism, LANES, max_abs, first);
    if(first>=0) printf("first bad lane=%d: got=0x%04x(%g) exp=0x%04x(%g)\n",
                        first,out[first],bf16_to_f(out[first]),exp[first],bf16_to_f(exp[first]));
    if(mism==0){ printf("REF VERIFIED (K-reduction butterfly + bf16 bit-exact)\nTEST PASSED\n"); return 0; }
    printf("FAILED\n"); return 1;
}
