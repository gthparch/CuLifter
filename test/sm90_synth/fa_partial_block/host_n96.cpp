#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
// Runs the LIFTED wgmma_n96_reduce kernel: HGMMA.64x96x16 QK GEMM + FA3
// reduce_max/scale_apply_exp2/reduce_sum over the n_dim=96 columns. Compares
// per-row (row_max,row_sum) vs a CPU reference computed from the same A,B.
extern "C" void wgmma_n96_reduce();

#define MM 64
#define KK 16
#define NN 96

static inline float h2f(uint16_t h){
    uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,out;
    if(e==0){ if(m==0) out=s<<31; else { e=1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; out=(s<<31)|((e+112)<<23)|(m<<13);} }
    else if(e==31){ out=(s<<31)|0x7f800000|(m<<13);} else out=(s<<31)|((e+112)<<23)|(m<<13);
    float f; memcpy(&f,&out,4); return f;
}
static inline uint16_t f2h(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000; int e=((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffff;
    if(e<=0) return (uint16_t)s;
    if(e>=31) return (uint16_t)(s|0x7c00);
    return (uint16_t)(s|(e<<10)|(m>>13));
}

int main() {
    static uint16_t hA[MM*KK], hB[NN*KK];
    static float hO[MM*2];
    for(int i=0;i<MM*KK;i++) hA[i]=f2h(((i*37%13)-6)*0.05f);
    for(int i=0;i<NN*KK;i++) hB[i]=f2h(((i*29%11)-5)*0.05f);
    memset(hO,0,sizeof(hO));

    launchKernel(wgmma_n96_reduce, 1, 128,
                 ArgAt{0x210}, (uint16_t*)hA,
                 ArgAt{0x218}, (uint16_t*)hB,
                 ArgAt{0x220}, (float*)hO);

    const float scale = 0.25f;
    const float scale_log2 = scale * 1.4426950408889634f;
    bool ok = true; float maxerr = 0.f;
    for (int m = 0; m < MM; m++) {
        // CPU reference: raw score s[n] = sum_k A[m,k]*B[n,k]; max over n; sum exp2.
        float rmax = -INFINITY;
        for (int n = 0; n < NN; n++) {
            float s = 0.f;
            for (int k = 0; k < KK; k++) s += h2f(hA[m*KK+k]) * h2f(hB[n*KK+k]);
            if (s > rmax) rmax = s;
        }
        float rsum = 0.f;
        for (int n = 0; n < NN; n++) {
            float s = 0.f;
            for (int k = 0; k < KK; k++) s += h2f(hA[m*KK+k]) * h2f(hB[n*KK+k]);
            rsum += exp2f(s*scale_log2 - rmax*scale_log2);
        }
        float gmax = hO[m*2+0], gsum = hO[m*2+1];
        float em = fabsf(gmax - rmax), es = fabsf(gsum - rsum);
        if (em > maxerr) maxerr = em;
        if (es > maxerr) maxerr = es;
        if ((em > 1e-2f || es > 1e-2f) && ok) {
            printf("FAIL row %d: max got=%.5f ref=%.5f | sum got=%.5f ref=%.5f\n",
                   m, gmax, rmax, gsum, rsum);
            ok = false;
        }
    }
    printf("max_err=%g\n", maxerr);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
