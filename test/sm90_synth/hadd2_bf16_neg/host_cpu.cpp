// Self-contained launch + verify for the bf16 per-lane register-negation path.
//
// Kernel: out[i] = __hadd2(__hneg2(a[i]), b[i])  (bf16x2)  -> (-a)+b per lane.
// The negate is an HFMA2.BF16_V2 with a `-R2` accumulator; this checks the
// bf16 branch of _get_val_half2 (per-lane sign-bit flip on <2 x bfloat>). The
// host computes (-a_lane + b_lane) in bf16 (round-to-nearest-even, the form
// nvcc/ptxas use) and checks the exact stored bf16 bits. No GPU needed.
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// f32 -> bf16, round-to-nearest-even (matches ptxas cvt.rn.bf16.f32)
static inline uint16_t f2bf(float f){
  uint32_t x; memcpy(&x,&f,4);
  if(((x>>23)&0xFF)==0xFF) return (uint16_t)((x>>16)&0xFFFF);   // inf/nan: pass top half
  uint32_t lsb=(x>>16)&1; uint32_t round=0x7FFF+lsb;
  x += round; return (uint16_t)((x>>16)&0xFFFF);
}
static inline float bf2f(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; memcpy(&f,&x,4); return f; }
// per-lane bf16 negate = flip sign bit
static inline uint16_t bfneg(uint16_t b){ return (uint16_t)(b ^ 0x8000); }

extern "C" void hadd2_bf16_neg();

int main(){
  constexpr int N=256;
  uint32_t *a=new uint32_t[N],*b=new uint32_t[N],*out=new uint32_t[N];
  memset(out,0xAA,N*sizeof(uint32_t));
  for(int i=0;i<N;i++){
    float a0=((i%17)-8)*0.5f;     // [-4,4]
    float a1=((i%13)-6)*0.25f;    // [-1.5,1.5]
    float b0=((i%11)-5)*0.5f;
    float b1=((i%7)-3)*0.5f;
    a[i]=(uint32_t)f2bf(a0) | ((uint32_t)f2bf(a1)<<16);
    b[i]=(uint32_t)f2bf(b0) | ((uint32_t)f2bf(b1)<<16);
  }
  launchKernel(hadd2_bf16_neg, (N+63)/64, 64, a, b, out, N);

  bool ok=true; int bad=0;
  for(int i=0;i<N;i++){
    uint16_t a0=a[i]&0xFFFF, a1=(a[i]>>16)&0xFFFF;
    uint16_t b0=b[i]&0xFFFF, b1=(b[i]>>16)&0xFFFF;
    // CORRECT: (-a + b) per lane in bf16 (per-lane sign flip on a)
    uint16_t r0=f2bf(bf2f(bfneg(a0)) + bf2f(b0));
    uint16_t r1=f2bf(bf2f(bfneg(a1)) + bf2f(b1));
    uint32_t ref=(uint32_t)r0 | ((uint32_t)r1<<16);
    if(out[i]!=ref){ if(bad<6) std::cout<<"FAIL i="<<i<<" got=0x"<<std::hex<<out[i]
        <<" exp=0x"<<ref<<std::dec<<" (a0="<<bf2f(a0)<<" b0="<<bf2f(b0)
        <<" a1="<<bf2f(a1)<<" b1="<<bf2f(b1)<<")\n"; bad++; ok=false; }
  }
  if(ok) std::cout<<"TEST PASSED\n"; else std::cout<<"TEST FAILED ("<<bad<<"/"<<N<<" bad)\n";
  delete[] a; delete[] b; delete[] out;
  return ok?0:1;
}
