// BF16 mainloop value-verify: launch the LIFTED warp-true kernel (32 lanes)
// and compare the 16x8 f32 output against an independent oracle that mirrors
// the HMMA.16816.F32.BF16 contraction (bf16-quantized A/B, f32 accumulate)
// summed over KT distinct k-tiles. The HMMA accumulates k=16 products per tile
// in the hardware's lane order; for a SINGLE warp the accumulation order of the
// k=16 inner products is fixed by the atom (and the oracle sums them in the
// same natural k order — for these magnitudes f32 accumulation is associative
// enough that we compare with a tight rel tol, and ALSO report exact match).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

static inline uint16_t f2bf_rn(float f){
  uint32_t x; memcpy(&x,&f,4);
  if(((x>>23)&0xFF)==0xFF){ if(f!=f) return 0x7FFF; return (uint16_t)((x>>16)&0xFFFF); }
  uint32_t lsb=(x>>16)&1u; x += 0x7FFFu+lsb; return (uint16_t)((x>>16)&0xFFFF);
}
static inline float bf2f(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; memcpy(&f,&x,4); return f; }

extern "C" void kern_bf16_mainloop();

int main(){
  const int KT = 5;            // 5 distinct k-tiles (k=80 total)
  const int M=16, Nn=8, K=16;
  // bf16 storage for A [KT][M][K], B [KT][K][Nn]
  uint16_t* A=new uint16_t[KT*M*K];
  uint16_t* B=new uint16_t[KT*K*Nn];
  float* D=new float[M*Nn];
  memset(D,0xCD,M*Nn*sizeof(float));    // canary

  // deterministic varied inputs, exactly representable-ish then bf16-quantized
  for(int i=0;i<KT*M*K;i++){
    float v = (((i*37+11)%97)-48) * 0.0625f;   // multiples of 1/16 in ~[-3,3]
    A[i]=f2bf_rn(v);
  }
  for(int i=0;i<KT*K*Nn;i++){
    float v = (((i*53+7)%89)-44) * 0.125f;      // multiples of 1/8 in ~[-5.5,5.5]
    B[i]=f2bf_rn(v);
  }

  // ORACLE: D[m][n] = sum_tile sum_k bf2f(A[tile][m][k]) * bf2f(B[tile][k][n])
  // accumulated in f32 (the HMMA accumulator is f32).
  float* ref=new float[M*Nn];
  for(int m=0;m<M;m++) for(int n=0;n<Nn;n++){
    float acc=0.f;
    for(int tile=0;tile<KT;tile++)
      for(int k=0;k<K;k++)
        acc += bf2f(A[tile*M*K + m*K + k]) * bf2f(B[tile*K*Nn + k*Nn + n]);
    ref[m*Nn+n]=acc;
  }

  // Launch lifted kernel: one warp (32 threads). Args: A, B, D, KT.
  launchKernel(kern_bf16_mainloop, 1, 32, A, B, D, KT);

  int bad=0, exact=0; double max_rel=0, max_abs=0; bool ok=true;
  for(int m=0;m<M;m++) for(int n=0;n<Nn;n++){
    float got=D[m*Nn+n], r=ref[m*Nn+n];
    if(got==r) exact++;
    double ae=fabs((double)got-(double)r);
    double re=ae/(fabs((double)r)+1e-30);
    if(ae>max_abs) max_abs=ae;
    if(re>max_rel) max_rel=re;
    if(re>1e-5 && ae>1e-3){
      if(bad<8) printf("FAIL [%d][%d] got=%.9g exp=%.9g (abs=%.3g rel=%.3g)\n",m,n,got,r,ae,re);
      bad++; ok=false;
    }
  }
  printf("exact=%d/%d max_abs=%.3g max_rel=%.3g\n", exact, M*Nn, max_abs, max_rel);
  if(ok) printf("TEST PASSED (16x8 bf16-mainloop, KT=%d, max_rel=%.3g)\n", KT, max_rel);
  else   printf("TEST FAILED (%d/%d bad)\n", bad, M*Nn);
  delete[] A; delete[] B; delete[] D; delete[] ref;
  return ok?0:1;
}
