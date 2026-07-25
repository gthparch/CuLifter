// Self-contained launch + verify for the half2 register-negation regression.
// Computes SiLu(d0)*d1 = d0/(1+exp(-d0)) * d1 on the host (f32) and checks the
// lifted x86 kernel matches within an f16 tolerance. No GPU required.
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

extern "C" void silu_half2_neg();

int main(){
  constexpr int N=512;
  uint16_t *d0=new uint16_t[N],*d1=new uint16_t[N],*out=new uint16_t[N];
  memset(out,0,N*sizeof(uint16_t));
  for(int i=0;i<N;i++){
    float a=((i%23)-11)*0.5f;       // span [-5.5,5.5] -> exercises both sigmoid tails+center
    float b=((i%13)-6)*0.5f;
    d0[i]=f2h(a); d1[i]=f2h(b);
  }
  launchKernel(silu_half2_neg, (N+255)/256, 256, d0, d1, out, N);
  bool ok=true; int bad=0;
  for(int i=0;i<N;i++){
    float x=h2f(d0[i]), y=h2f(d1[i]);
    float sig=1.0f/(1.0f+expf(-x)); float ref=x*sig*y;
    float o=h2f(out[i]);
    // f16 tolerance (the half-compute sigmoid + half output)
    float tol = 0.02f*fabsf(ref) + 0.02f;
    if(fabsf(o-ref)>tol){ if(bad<5) std::cout<<"FAIL i="<<i<<" d0="<<x<<" d1="<<y
        <<" got "<<o<<" exp "<<ref<<"\n"; bad++; ok=false; }
  }
  if(ok) std::cout<<"TEST PASSED\n"; else std::cout<<"TEST FAILED ("<<bad<<"/"<<N<<" bad)\n";
  delete[] d0; delete[] d1; delete[] out;
  return ok?0:1;
}
