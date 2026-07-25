#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"
extern "C" void tma_sw128_k();

#ifndef KK
#define KK 64
#endif
#define MM 64
#define NN 32

static inline float bf2f(uint16_t h){ uint32_t x=((uint32_t)h)<<16; float f; memcpy(&f,&x,4); return f; }
static inline uint16_t f2bf(float f){ uint32_t x; memcpy(&x,&f,4); return (uint16_t)((x+0x8000)>>16); }

int main() {
    static uint16_t hA[MM*KK], hB[NN*KK];
    static float outD[MM*NN];
    for(int i=0;i<MM*KK;i++) hA[i]=f2bf(((i*37%13)-6)*0.05f);
    for(int i=0;i<NN*KK;i++) hB[i]=f2bf(((i*29%11)-5)*0.05f);
    memset(outD,0,sizeof(outD));

    // Descriptor block at const[0x198]; descA at base+0x30, descB at base+0xf0.
    alignas(64) static uint8_t descblk[0x200];
    memset(descblk,0,sizeof(descblk));
    SlifterTmaDesc* descA = reinterpret_cast<SlifterTmaDesc*>(descblk + 0x30);
    SlifterTmaDesc* descB = reinterpret_cast<SlifterTmaDesc*>(descblk + 0xf0);
    *descA = SlifterTmaDesc{};
    descA->global_base=hA; descA->rows=MM; descA->cols_k=KK;
    descA->elem_bytes=2; descA->swizzle=3; descA->ld=KK; descA->tile_base_off=0x400;
    *descB = SlifterTmaDesc{};
    descB->global_base=hB; descB->rows=NN; descB->cols_k=KK;
    descB->elem_bytes=2; descB->swizzle=3; descB->ld=KK; descB->tile_base_off=0x2400;

    uint64_t desc_param = reinterpret_cast<uint64_t>(descblk);

    launchKernel(tma_sw128_k, 1, 128,
                 ArgAt{0x198}, (uint64_t)desc_param,
                 ArgAt{0x3c0}, (float*)outD);

    float maxerr=0; int nbad=0;
    for(int m=0;m<MM;m++) for(int n=0;n<NN;n++){
        float s=0; for(int k=0;k<KK;k++) s+=bf2f(hA[m*KK+k])*bf2f(hB[n*KK+k]);
        float g=outD[m*NN+n], e=fabsf(g-s);
        if(e>maxerr)maxerr=e;
        if(e>1e-2f){ if(nbad<10) printf("D[%d][%d] got=%.4f ref=%.4f\n",m,n,g,s); nbad++; }
    }
    printf("KK=%d  raw-D max_err=%g  bad=%d/%d\n", KK, maxerr, nbad, MM*NN);
    printf(maxerr<1e-2f?"TEST PASSED\n":"TEST FAILED\n");
    return maxerr<1e-2f?0:1;
}
