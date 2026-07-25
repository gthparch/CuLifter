// nvjpeg::ycbcr_to_format_kernel_roi<420(2), false, RGB(3), LaunchParams<32,8>>
// Same YCbCr 4:2:0 conversion math as the BGR variant, output plane order R,G,B.
// Output is uint8 derived from FP (FFMA + clamp + truncate); compared with +/-1
// TOLERANCE (cmp_u8_tol tol=1) so a 1-ULP rounding difference between the lifted
// FP and the host fmaf reference does not spuriously fail.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"

extern "C" void
_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t2ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

static uint8_t ct(float v){ v=v>=0?v:0; v=v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }

int main() {
    constexpr int W=16,H=8,CW=W/2,CH=H/2;
    uint8_t Y[H*W],Cb[CH*CW],Cr[CH*CW], oR[H*W],oG[H*W],oB[H*W], rR[H*W],rG[H*W],rB[H*W];
    for(int r=0;r<H;++r)for(int c=0;c<W;++c) Y[r*W+c]=(uint8_t)(r*23+c*11+7);
    for(int r=0;r<CH;++r)for(int c=0;c<CW;++c){ Cb[r*CW+c]=(uint8_t)(r*31+c*17+40); Cr[r*CW+c]=(uint8_t)(r*13+c*29+200); }
    for(int r=0;r<H;++r)for(int c=0;c<W;++c){
        float Yf=Y[r*W+c], Cbm=(float)Cb[(r>>1)*CW+(c>>1)]-128, Crm=(float)Cr[(r>>1)*CW+(c>>1)]-128;
        rB[r*W+c]=ct(fmaf(Cbm,1.7719999551773071289f,Yf));
        rG[r*W+c]=ct(fmaf(Crm,-0.71413630247116088867f,fmaf(Cbm,-0.34413629770278930664f,Yf)));
        rR[r*W+c]=ct(fmaf(Crm,1.4019999504089355469f,Yf));
    }
    for(int i=0;i<H*W;++i) oR[i]=oG[i]=oB[i]=0xCC;
    launchKernel2D(
        _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t2ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        Dim3{(W+31)/32,(H+7)/8}, Dim3{32,8},
        ArgAt{0x210},(uint8_t*)Y, ArgAt{0x218},(uint8_t*)Cb, ArgAt{0x220},(uint8_t*)Cr,
        ArgAt{0x230},(uint64_t)W, ArgAt{0x238},(uint64_t)CW, ArgAt{0x240},(uint64_t)CW,
        ArgAt{0x250},(int)W, ArgAt{0x254},(int)H,
        ArgAt{0x258},(uint8_t*)oR, ArgAt{0x260},(uint8_t*)oG, ArgAt{0x268},(uint8_t*)oB,
        ArgAt{0x278},(uint64_t)W, ArgAt{0x280},(uint64_t)W, ArgAt{0x288},(uint64_t)W,
        ArgAt{0x298},(int)0, ArgAt{0x29C},(int)0);
    int bad = cmp_u8_tol(oR,rR,H*W,1,"R")+cmp_u8_tol(oG,rG,H*W,1,"G")+cmp_u8_tol(oB,rB,H*W,1,"B");
    printf("=== ycbcr->RGB 420 (tol +/-1): px0 RGB=(%u,%u,%u) ref=(%u,%u,%u) ===\n",
           oR[0],oG[0],oB[0],rR[0],rG[0],rB[0]);
    return report(bad, 3*H*W);
}
