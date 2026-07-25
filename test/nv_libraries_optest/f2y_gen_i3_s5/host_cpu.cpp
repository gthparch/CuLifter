// AUTO-GENERATED (gen_format_to_ycbcr.py 3 5) infmt=3 subsamp=5
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"
extern "C" void _ZN6nvjpeg22format_to_ycbcr_kernelIL19nvjpegInputFormat_t3EL25nvjpegChromaSubsampling_t5ENS_15ColorModelToYUV12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();
static uint8_t ct(float v){ v=v>=0?v:0; v=v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }
static float Yf (float R,float G,float B){ return fmaf(B,0.11400000005960464478f, fmaf(R,0.29899999499320983887f, G*0.58700001239776611328f)); }
static float Cbf(float R,float G,float B){ return fmaf(B,0.5f, fmaf(R,-0.16873589158058166504f, -(G*0.33126410841941833496f))) + 128.0f; }
static float Crf(float R,float G,float B){ return fmaf(B,-0.081312410533428192139f, fmaf(R,0.5f, -(G*0.41868758201599121094f))) + 128.0f; }
int main(){
    const int W=16,H=8;
    uint8_t P0[8*16],P1[8*16],P2[8*16];
    for(int r=0;r<8;++r)for(int c=0;c<16;++c){ P0[r*16+c]=(uint8_t)(r*23+c*11+7); P1[r*16+c]=(uint8_t)(r*31+c*17+40); P2[r*16+c]=(uint8_t)(r*13+c*29+200); }
    uint8_t oY[8*16],oCb[4*4],oCr[4*4],rY[8*16],rCb[4*4],rCr[4*4];
    for(int i=0;i<8*16;++i)oY[i]=0xCC; for(int i=0;i<4*4;++i){oCb[i]=0xCC;oCr[i]=0xCC;}
    for(int r=0;r<8;++r)for(int c=0;c<16;++c){ float R=(float)P0[r*16+c], G=(float)P1[r*16+c], B=(float)P2[r*16+c]; rY[r*16+c]=ct(Yf(R,G,B)); }
    for(int cy=0;cy<4;++cy)for(int cx=0;cx<4;++cx){
        float sb=0,sr=0;
        for(int dy=0;dy<2;++dy)for(int dx=0;dx<4;++dx){ int r=2*cy+dy,c=4*cx+dx; float R=(float)P0[r*16+c], G=(float)P1[r*16+c], B=(float)P2[r*16+c]; sb+=(float)ct(Cbf(R,G,B)); sr+=(float)ct(Crf(R,G,B)); }
        rCb[cy*4+cx]=ct(sb*0.125f); rCr[cy*4+cx]=ct(sr*0.125f);
    }
    launchKernel2D(_ZN6nvjpeg22format_to_ycbcr_kernelIL19nvjpegInputFormat_t3EL25nvjpegChromaSubsampling_t5ENS_15ColorModelToYUV12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_, Dim3{1,1}, Dim3{4,4},
        ArgAt{0x210},(uint8_t*)P0, ArgAt{0x218},(uint8_t*)P1, ArgAt{0x220},(uint8_t*)P2,
        ArgAt{0x230},(uint64_t)16, ArgAt{0x238},(uint64_t)16, ArgAt{0x240},(uint64_t)16,
        ArgAt{0x250},(int)W, ArgAt{0x254},(int)H,
        ArgAt{0x258},(uint8_t*)oY, ArgAt{0x260},(uint8_t*)oCb, ArgAt{0x268},(uint8_t*)oCr,
        ArgAt{0x278},(uint64_t)16, ArgAt{0x280},(uint64_t)4, ArgAt{0x288},(uint64_t)4,
        ArgAt{0x298},(int)0, ArgAt{0x29C},(int)0);
    int bad=cmp_u8_tol(oY,rY,8*16,1,"Y")+cmp_u8_tol(oCb,rCb,4*4,1,"Cb")+cmp_u8_tol(oCr,rCr,4*4,1,"Cr");
    return report(bad,8*16+2*4*4);
}
