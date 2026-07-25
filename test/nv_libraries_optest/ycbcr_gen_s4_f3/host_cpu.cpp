// AUTO-GENERATED (gen_ycbcr.py 4 3) — ycbcr_to_format subsamp=4 fmt=3
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"
extern "C" void _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t4ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();
static uint8_t ct(float v){ v=v>=0?v:0; v=v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }
int main(){
    const int W=16,H=8;
    uint8_t Y[H*W]; for(int r=0;r<H;++r)for(int c=0;c<W;++c) Y[r*W+c]=(uint8_t)(r*23+c*11+7);
    uint8_t Cb[8*4], Cr[8*4];
    for(int r=0;r<8;++r)for(int c=0;c<4;++c){ Cb[r*4+c]=(uint8_t)(r*31+c*17+40); Cr[r*4+c]=(uint8_t)(r*13+c*29+200); }
    uint8_t R0[8*16],R1[8*16],R2[8*16];
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){
        float Yf=Y[r*W+c];
        float Cbm=(float)Cb[((r>>0)*4 + (c>>2))]-128, Crm=(float)Cr[((r>>0)*4 + (c>>2))]-128;
        float B=fmaf(Cbm,1.7719999551773071289f,Yf);
        float G=fmaf(Crm,-0.71413630247116088867f,fmaf(Cbm,-0.34413629770278930664f,Yf));
        float R=fmaf(Crm,1.4019999504089355469f,Yf);
        R0[r*16+c]=ct(R); R1[r*16+c]=ct(G); R2[r*16+c]=ct(B);
    }
    uint8_t O0[8*16],O1[8*16],O2[8*16]; for(int i=0;i<8*16;++i)O0[i]=O1[i]=O2[i]=0xCC;
    launchKernel2D(_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t4ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_, Dim3{(W+31)/32,(H+7)/8}, Dim3{32,8},
        ArgAt{0x210},(uint8_t*)Y,
        ArgAt{0x218},(uint8_t*)Cb, ArgAt{0x220},(uint8_t*)Cr,
        ArgAt{0x238},(uint64_t)4, ArgAt{0x240},(uint64_t)4,
        ArgAt{0x230},(uint64_t)W,
        ArgAt{0x250},(int)W, ArgAt{0x254},(int)H,
        ArgAt{0x258},(uint8_t*)O0, ArgAt{0x260},(uint8_t*)O1, ArgAt{0x268},(uint8_t*)O2,
        ArgAt{0x278},(uint64_t)16, ArgAt{0x280},(uint64_t)16, ArgAt{0x288},(uint64_t)16,
        ArgAt{0x298},(int)0, ArgAt{0x29C},(int)0);
    int bad=cmp_u8_tol(O0,R0,8*16,1,"c0")+cmp_u8_tol(O1,R1,8*16,1,"c1")+cmp_u8_tol(O2,R2,8*16,1,"c2");
    return report(bad,3*8*16);
}
