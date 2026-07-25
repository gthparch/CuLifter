#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"        // -> kernel_wrapper_75.h
#include "kernel_wrapper_warp.h"   // x86warp fiber backend (launchKernel_warp)
// Runs the LIFTED producer-phase-1 ring pipeline under the x86warp FIBER backend
// (the model the FA3 paged_split kernels run under) and verifies vs a CPU oracle.
extern "C" void ring_prime();
#define D 16
#define SQ 32
#define SK 64
int main() {
    float Q[SQ*D], K[SK*D], Vv[SK*D], O[SQ*D];
    for (int i=0;i<SQ*D;i++) Q[i]  = (float)(((i*7+5)%13)-6)*0.1f;
    for (int i=0;i<SK*D;i++) K[i]  = (float)(((i*5+3)%11)-5)*0.1f;
    for (int i=0;i<SK*D;i++) Vv[i] = (float)(((i*3+1)%9)-4)*0.1f;
    memset(O,0,sizeof O);
    launchKernel_warp(ring_prime, 1, 64, Q, K, Vv, O, SK);
    const float scale=0.25f; bool ok=true; float maxerr=0;
    for (int i=0;i<SQ&&ok;i++){ float s[SK],mx=-1e30f;
        for(int j=0;j<SK;j++){float d=0;for(int k=0;k<D;k++)d+=Q[i*D+k]*K[j*D+k];s[j]=d*scale;if(s[j]>mx)mx=s[j];}
        float den=0;for(int j=0;j<SK;j++){s[j]=expf(s[j]-mx);den+=s[j];}
        for(int k=0;k<D;k++){float o=0;for(int j=0;j<SK;j++)o+=s[j]*Vv[j*D+k];o/=den;float e=fabsf(o-O[i*D+k]);if(e>maxerr)maxerr=e;if(e>1e-3f){printf("FAIL %d %d got %f exp %f\n",i,k,O[i*D+k],o);ok=false;break;}}}
    printf("max_err=%g\n",maxerr); printf(ok?"TEST PASSED\n":"TEST FAILED\n");
    return ok?0:1;
}
