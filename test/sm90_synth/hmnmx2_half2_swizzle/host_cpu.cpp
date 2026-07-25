// Self-contained launch + verify for the HMNMX2 .H0_H0 swizzle-drop bug.
//
// Kernel: out[i] = __hmax2(a[i], __half2half2(s[i]))
//   -> HMNMX2 Rd, Ra, Rb.H0_H0, !PT   (broadcast s to both lanes, then max)
//   correct result = {max(a.lo, s), max(a.hi, s)}.
//
// The bug dropped the .H0_H0 swizzle, so the lifted kernel used Rb verbatim =
// {s, 0.0} (the scalar is a 16-bit zero-extended load), giving a WRONG high
// lane {max(a.lo, s), max(a.hi, 0.0)}. We pick negative s and a high lane
// below 0 so the two differ, and check exact f16 bits. No GPU needed.
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

extern "C" void hmnmx2_half2_swizzle();

int main(){
  constexpr int N=256;
  uint32_t *a=new uint32_t[N],*out=new uint32_t[N];
  uint16_t *s=new uint16_t[N];
  memset(out,0xAA,N*sizeof(uint32_t));
  for(int i=0;i<N;i++){
    float lo=((i%19)-9)*0.5f;     // [-4.5, 4.5]
    float hi=((i%23)-15)*0.5f;    // skewed negative -> exercises max(hi, s) vs max(hi, 0)
    a[i]=(uint32_t)f2h(lo) | ((uint32_t)f2h(hi)<<16);
    float sc=((i%7)-5)*0.5f;      // mostly negative scalar broadcast
    s[i]=f2h(sc);
  }
  launchKernel(hmnmx2_half2_swizzle, (N+63)/64, 64, a, s, out, N);

  bool ok=true; int bad=0;
  for(int i=0;i<N;i++){
    float alo=h2f(a[i]&0xFFFF), ahi=h2f((a[i]>>16)&0xFFFF);
    float sc=h2f(s[i]);
    // CORRECT: broadcast sc to BOTH lanes, then per-lane max
    uint16_t r0=f2h(fmaxf(alo, sc));
    uint16_t r1=f2h(fmaxf(ahi, sc));
    uint32_t ref=(uint32_t)r0 | ((uint32_t)r1<<16);
    if(out[i]!=ref){ if(bad<6) std::cout<<"FAIL i="<<i<<" got=0x"<<std::hex<<out[i]
        <<" exp=0x"<<ref<<std::dec<<" (alo="<<alo<<" ahi="<<ahi<<" s="<<sc<<")\n"; bad++; ok=false; }
  }
  if(ok) std::cout<<"TEST PASSED\n"; else std::cout<<"TEST FAILED ("<<bad<<"/"<<N<<" bad)\n";
  delete[] a; delete[] s; delete[] out;
  return ok?0:1;
}
