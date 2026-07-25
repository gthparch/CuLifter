// Rigorous F2FP.BF16.F32.PACK_AB value-verify: runs the LIFTED kernel and
// compares EVERY output bf16 lane bit-for-bit against an independent host
// round-to-nearest-even oracle. Inputs are constructed to STRESS the RN-even
// tie boundary (dropped low-16 bits == 0x8000, where the mantissa bit-17 lsb
// decides round-up vs round-to-even). A truncation/wrong-rounding lifter bug
// would diverge on the tie cases. The lifted x86 kernel is exercised; the
// oracle is the spec.
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

// f32 bits -> bf16 bits, round-to-nearest-even (the cvt.rn.bf16.f32 / cutlass
// NumericConverter<bfloat16_t,float,round_to_nearest> spec).
static inline uint16_t f2bf_rn(uint32_t x){
  if(((x>>23)&0xFF)==0xFF){                 // inf/nan
    float f; memcpy(&f,&x,4);
    if(f!=f) return 0x7FFF;                  // NaN -> canonical 0x7FFF
    return (uint16_t)((x>>16)&0xFFFF);       // inf -> top half
  }
  uint32_t lsb=(x>>16)&1u;
  uint32_t rounded = x + 0x7FFFu + lsb;
  return (uint16_t)((rounded>>16)&0xFFFF);
}
static inline uint32_t f2u(float f){ uint32_t x; memcpy(&x,&f,4); return x; }

extern "C" void kern_f2fp_bf16_pack();

int main(){
  constexpr int N=512;
  float* in=new float[N*2];
  uint32_t* out=new uint32_t[N];
  memset(out,0xCD,N*sizeof(uint32_t));   // canary

  for(int i=0;i<N;i++){
    if(i<128){
      // exact ties: low 16 bits == 0x8000; kept lsb (bit 16) alternates
      uint32_t exp = (127u + (i&3)) << 23;
      uint32_t mant_hi = ((uint32_t)(i>>2) & 0x7F) << 16;
      uint32_t a = exp | mant_hi | 0x8000u;            // low16 == 0x8000 -> TIE
      uint32_t b = exp | mant_hi | 0x8000u | (1u<<16); // tie, opposite lsb
      if(i&1){ a|=0x80000000u; }
      float fa,fb; memcpy(&fa,&a,4); memcpy(&fb,&b,4);
      in[i*2]=fa; in[i*2+1]=fb;
    } else if(i<160){
      uint32_t exp=(127u+(i&7))<<23;
      uint32_t a=exp | (((uint32_t)i&0x3F)<<16) | 0x7FFFu; // just-below -> round DOWN
      uint32_t b=exp | (((uint32_t)i&0x3F)<<16) | 0x8001u; // just-above -> round UP
      float fa,fb; memcpy(&fa,&a,4); memcpy(&fb,&b,4);
      in[i*2]=fa; in[i*2+1]=fb;
    } else {
      in[i*2]   = ((i%37)-18) * 0.137f * (1.0f + (i%5));
      in[i*2+1] = ((i%23)-11) * 9.31e3f / (1.0f + (i%3));
    }
  }
  in[2*200]=INFINITY;  in[2*200+1]=-INFINITY;
  in[2*201]=NAN;       in[2*201+1]=0.0f;
  in[2*202]=-0.0f;     in[2*202+1]=1.0f/3.0f;

  launchKernel(kern_f2fp_bf16_pack, (N+255)/256, 256, in, out, N);

  int bad=0, ties=0; bool ok=true;
  for(int i=0;i<N;i++){
    uint16_t ra=f2bf_rn(f2u(in[i*2]));
    uint16_t rb=f2bf_rn(f2u(in[i*2+1]));
    uint32_t ref=(uint32_t)ra | ((uint32_t)rb<<16);
    if(((f2u(in[i*2])&0xFFFF)==0x8000)) ties++;
    if(((f2u(in[i*2+1])&0xFFFF)==0x8000)) ties++;
    if(out[i]!=ref){
      if(bad<8) printf("FAIL i=%d got=0x%08X exp=0x%08X (a=%.9g b=%.9g  abits=0x%08X bbits=0x%08X)\n",
                       i,out[i],ref,in[i*2],in[i*2+1],f2u(in[i*2]),f2u(in[i*2+1]));
      bad++; ok=false;
    }
  }
  printf("ties exercised: %d\n", ties);
  if(ok) printf("TEST PASSED (%d lanes bit-exact, %d ties)\n", N*2, ties);
  else   printf("TEST FAILED (%d/%d bad)\n", bad, N);
  delete[] in; delete[] out;
  return ok?0:1;
}
