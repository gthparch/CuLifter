#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <algorithm>
#include "kernel_wrapper.h"
extern "C" void kern_hmma_ldsm_tf32_val();
static float snap_tf32(float f){uint32_t x;std::memcpy(&x,&f,4);x&=0xFFFFE000u;float r;std::memcpy(&r,&x,4);return r;}
int main(){
    constexpr int M=16,N=8,K=8;
    std::vector<float> A(M*K),B(K*N),C(M*N),D(M*N,-7.0f);
    uint32_t s=0x1D32F32u;
    auto nxt=[&](){s=s*1664525u+1013904223u;return s;};
    auto rnd=[&](float sc){return sc*(((int)(nxt()>>16&0x3FF)-512)/256.0f);};
    for(int i=0;i<M*K;++i)A[i]=snap_tf32(rnd(1.0f));
    for(int i=0;i<K*N;++i)B[i]=snap_tf32(rnd(1.0f));
    for(int i=0;i<M*N;++i)C[i]=rnd(2.0f);
    launchKernel(kern_hmma_ldsm_tf32_val,1,32,
        ArgAt{0x210},A.data(),ArgAt{0x218},B.data(),ArgAt{0x220},C.data(),ArgAt{0x228},D.data());
    int nbad=0,nbadrows=0; double maxd=0;
    printf("per-row:\n");
    for(int m=0;m<M;++m){int rb=0;double rmax=0;
        for(int n=0;n<N;++n){double acc=C[m*N+n];
            for(int k=0;k<K;++k)acc+=(double)A[m*K+k]*(double)B[k*N+n];
            double got=D[m*N+n],diff=std::fabs(got-acc),rel=diff/std::max(std::fabs(acc),1.0);
            rmax=std::max(rmax,rel);maxd=std::max(maxd,rel);if(rel>1e-3){rb++;nbad++;}}
        int gg=m%8;printf("  row %2d (g=%d g&2=%d): %s maxrel=%.4g\n",m,gg,(gg&2)?1:0,rb?"BAD ":"GOOD",rmax);
        if(rb)nbadrows++;}
    if(!nbad){printf("REF VERIFIED (max_rel=%g)\nTEST PASSED\n",maxd);return 0;}
    printf("FAILED: %d/%d elems %d/%d rows max_rel=%g\n",nbad,M*N,nbadrows,M,maxd);return 1;
}
