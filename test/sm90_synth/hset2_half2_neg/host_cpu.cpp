// Self-contained launch + verify for the HSET2 half2 register-negation bug.
//
// Kernel: out[i] = __heq2(__hneg2(a[i]), b[i])  ->  HSET2.BF.EQ.AND Rd,-Ra,Rb,PT
// per-lane result mask is 0xFFFF (equal) / 0x0000 (not). The host computes the
// SAME per-lane (-a_lane == b_lane) compare in f16 and checks the exact mask
// bits. No GPU needed.
//
// The bug: `-Ra` was an i32 two's-complement negate of the packed f16x2 word
// (e.g. -0x38003800 = 0xC7FFC800 => lane0 = -8.0, not -0.5), so the lane
// values fed to the compare were wrong => wrong mask. Inputs below are chosen
// so that for MANY lanes the correct per-lane (-a == b) is TRUE, but the
// word-negated value is NOT equal to b => the buggy lift produces 0x0000
// where 0xFFFF is required (and vice-versa).
#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

static inline uint16_t f2h(float f){uint32_t x;memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000;
  int e=((x>>23)&0xFF)-127+15;uint32_t m=x&0x7FFFFF;
  if(((x>>23)&0xFF)==0xFF)return (uint16_t)(s|0x7C00|((m>>13)?0x200:0));
  if(e<=0){if(e<-10)return (uint16_t)s;m|=0x800000;int sh=14-e;uint32_t hf=1u<<(sh-1);
    uint32_t dr=m&((1u<<sh)-1);m>>=sh;if(dr>hf||(dr==hf&&(m&1)))m++;return (uint16_t)(s|m);}
  if(e>=31)return (uint16_t)(s|0x7C00);uint32_t dr=m&0x1FFF;m>>=13;
  if(dr>0x1000||(dr==0x1000&&(m&1)))m++;if(m>0x3FF){m=0;e++;}
  if(e>=31)return (uint16_t)(s|0x7C00);return (uint16_t)(s|(e<<10)|m);}
static inline float h2f(uint16_t h){uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,r;
  if(e==0){if(m==0)r=s;else{e=1;while(!(m&0x400)){m<<=1;e--;}m&=0x3FF;r=s|((e+112)<<23)|(m<<13);}}
  else if(e==31)r=s|0x7F800000|(m<<13);else r=s|((e+112)<<23)|(m<<13);
  float f;memcpy(&f,&r,4);return f;}

// per-lane f16 negate = flip sign bit (the CORRECT semantics of -Ra on f16x2)
static inline uint16_t hneg(uint16_t h){return (uint16_t)(h ^ 0x8000);}

extern "C" void hset2_half2_neg();

int main(){
  constexpr int N=256;
  // pack two f16 lanes into one u32 (lane0 = low 16 bits)
  uint32_t *a=new uint32_t[N],*b=new uint32_t[N],*out=new uint32_t[N];
  memset(out,0xAA,N*sizeof(uint32_t));   // canary (neither 0x0000 nor 0xFFFF)
  for(int i=0;i<N;i++){
    // lane0: a small positive, b = -a -> (-a == b) TRUE on correct lift.
    float a0=((i%17)-8)*0.5f;     // [-4, 4]
    float a1=((i%11)-5)*0.25f;    // [-1.25, 1.25]
    // b lanes set so the per-lane (-a == b) comparison is a MIX of T/F:
    float b0=-a0;                 // equal under correct per-lane negate
    float b1=(i&1)? -a1 : a1;     // half equal, half not
    uint16_t ah=f2h(a0), al=f2h(a1), bh=f2h(b0), bl=f2h(b1);
    a[i]=(uint32_t)ah | ((uint32_t)al<<16);
    b[i]=(uint32_t)bh | ((uint32_t)bl<<16);
  }
  launchKernel(hset2_half2_neg, (N+63)/64, 64, a, b, out, N);

  bool ok=true; int bad=0;
  // NOTE on the TRUE mask value: GPU HSET2.BF writes 0xFFFF, but the lifter
  // (BUG-033) intentionally emits half(1.0)=0x3C00 for a TRUE lane so a
  // downstream HFMA2/HMUL2 FP multiply by the mask works on IEEE x86 (0xFFFF
  // is NaN there). So the ORACLE uses 0x3C00/0x0000 — this microbench isolates
  // the per-lane NEGATION correctness (which lane is true/false), independent
  // of the mask-value convention.
  for(int i=0;i<N;i++){
    uint16_t a0=a[i]&0xFFFF, a1=(a[i]>>16)&0xFFFF;
    uint16_t b0=b[i]&0xFFFF, b1=(b[i]>>16)&0xFFFF;
    // CORRECT reference: per-lane (-a == b) -> 0x3C00 (true) / 0x0000 (false)
    uint16_t r0=(h2f(hneg(a0))==h2f(b0))?0x3C00:0x0000;
    uint16_t r1=(h2f(hneg(a1))==h2f(b1))?0x3C00:0x0000;
    uint32_t ref=(uint32_t)r0 | ((uint32_t)r1<<16);
    if(out[i]!=ref){ if(bad<6) std::cout<<"FAIL i="<<i<<" got=0x"<<std::hex<<out[i]
        <<" exp=0x"<<ref<<std::dec<<" (a0="<<h2f(a0)<<" b0="<<h2f(b0)
        <<" a1="<<h2f(a1)<<" b1="<<h2f(b1)<<")\n"; bad++; ok=false; }
  }
  if(ok) std::cout<<"TEST PASSED\n"; else std::cout<<"TEST FAILED ("<<bad<<"/"<<N<<" bad)\n";
  delete[] a; delete[] b; delete[] out;
  return ok?0:1;
}
