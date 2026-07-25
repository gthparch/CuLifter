// host_cpu.cpp — x86 exec-correctness for nvjpeg planar-BGR -> planar YCbCr 4:2:0.
//
// nvjpeg::format_to_ycbcr_kernel<InputFormat_t4(planar BGR), 420(2), ColorModelToYUV,
//                                LaunchParams<32,8>>
//   (nvjpegImage_t in, NppiSize inSize, nvjpegImage_t out, NppiSize roi)
//
// Param layout (from EIATTR_KPARAM_INFO, abs = 0x210 + Offset):
//   Ord0 nvjpegImage_t in   @0x210 (ch0..2 @0x210/0x218/0x220, pitch0..2 @0x230/0x238/0x240)
//   Ord1 NppiSize   inSize  @0x250 (w@0x250 h@0x254)
//   Ord2 nvjpegImage_t out  @0x258 (ch0..2 @0x258/0x260/0x268, pitch0..2 @0x278/0x280/0x288)
//   Ord3 NppiSize   roi     @0x298
//
// Decoded from full SASS. Each thread handles a 2x2 pixel block:
//   row pair (2*R, 2*R+1), col pair (2*C, 2*C+1)  (R=ctaid.y*8+tid.y, C=ctaid.x*32+tid.x)
// edge pixels are clamped to (w-1,h-1) (SEL with width/height-1). 4:2:0 => one
// Cb and one Cr per block (chroma is the average of the four pixels).
//
// INPUT is PLANAR BGR (verified from coefficients): ch0=B, ch1=G, ch2=R.
//   Y  = 0.299*R + 0.587*G + 0.114*B
//   Cb = -0.168736*R - 0.331264*G + 0.5*B + 128
//   Cr =  0.5*R - 0.418688*G - 0.081312*B + 128
//   each clamped [0,255], F2I.U32.TRUNC.
// OUTPUT: Y plane full-res (4 values, out.ch0); Cb (out.ch1) and Cr (out.ch2) at
//   half res = AVERAGE of the 4 block chroma values: (c00+c01+c10+c11) * 0.25.
//   (SASS: FADD sum of 4, then `FMUL.D2 sum, 0.5` => sum*0.5/2 = sum*0.25.)
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"

extern "C" void
_ZN6nvjpeg22format_to_ycbcr_kernelIL19nvjpegInputFormat_t4EL25nvjpegChromaSubsampling_t2ENS_15ColorModelToYUV12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

static uint8_t ct(float v){ v = v>=0?v:0; v = v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }
static float Yf (float R,float G,float B){ return fmaf(B,0.11400000005960464478f, fmaf(R,0.29899999499320983887f, G*0.58700001239776611328f)); }
static float Cbf(float R,float G,float B){ return fmaf(B,0.5f, fmaf(R,-0.16873589158058166504f, -(G*0.33126410841941833496f))) + 128.0f; }
static float Crf(float R,float G,float B){ return fmaf(B,-0.081312410533428192139f, fmaf(R,0.5f, -(G*0.41868758201599121094f))) + 128.0f; }

int main() {
    constexpr int W = 16, H = 8;            // even dims -> 4:2:0
    constexpr int CW = W/2, CH = H/2;
    uint8_t B[H*W], G[H*W], R[H*W];          // planar input (ch0=B, ch1=G, ch2=R)
    uint8_t oY[H*W], oCb[CH*CW], oCr[CH*CW];
    uint8_t rY[H*W], rCb[CH*CW], rCr[CH*CW];

    for (int r=0;r<H;++r) for (int c=0;c<W;++c) {
        B[r*W+c] = (uint8_t)(r*23 + c*11 + 7);
        G[r*W+c] = (uint8_t)(r*31 + c*17 + 40);
        R[r*W+c] = (uint8_t)(r*13 + c*29 + 200);
    }
    // independent host reference
    for (int r=0;r<H;++r) for (int c=0;c<W;++c)
        rY[r*W+c] = ct(Yf((float)R[r*W+c],(float)G[r*W+c],(float)B[r*W+c]));
    for (int cy=0;cy<CH;++cy) for (int cx=0;cx<CW;++cx) {
        float sb=0, sr=0;
        for (int dy=0;dy<2;++dy) for (int dx=0;dx<2;++dx) {
            int r=2*cy+dy, c=2*cx+dx;
            float Rf=(float)R[r*W+c], Gf=(float)G[r*W+c], Bf=(float)B[r*W+c];
            sb += (float)ct(Cbf(Rf,Gf,Bf));   // each block chroma is clamped+F2I, then averaged
            sr += (float)ct(Crf(Rf,Gf,Bf));
        }
        rCb[cy*CW+cx] = ct(sb*0.25f);
        rCr[cy*CW+cx] = ct(sr*0.25f);
    }
    for (int i=0;i<H*W;++i) oY[i]=0xCC;
    for (int i=0;i<CH*CW;++i){ oCb[i]=0xCC; oCr[i]=0xCC; }

    launchKernel2D(
        _ZN6nvjpeg22format_to_ycbcr_kernelIL19nvjpegInputFormat_t4EL25nvjpegChromaSubsampling_t2ENS_15ColorModelToYUV12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        // No internal bounds-EXIT: launch EXACTLY CW x CH threads, one per 2x2 block.
        Dim3{1,1}, Dim3{CW,CH},
        ArgAt{0x210},(uint8_t*)B, ArgAt{0x218},(uint8_t*)G, ArgAt{0x220},(uint8_t*)R,
        ArgAt{0x230},(uint64_t)W, ArgAt{0x238},(uint64_t)W, ArgAt{0x240},(uint64_t)W,
        ArgAt{0x250},(int)W,      ArgAt{0x254},(int)H,
        ArgAt{0x258},(uint8_t*)oY, ArgAt{0x260},(uint8_t*)oCb, ArgAt{0x268},(uint8_t*)oCr,
        ArgAt{0x278},(uint64_t)W, ArgAt{0x280},(uint64_t)CW, ArgAt{0x288},(uint64_t)CW,
        ArgAt{0x298},(int)0,      ArgAt{0x29C},(int)0);

    int badY  = cmp_u8_tol(oY, rY, H*W, 1, "Y");
    int badCb = cmp_u8_tol(oCb,rCb,CH*CW,1,"Cb");
    int badCr = cmp_u8_tol(oCr,rCr,CH*CW,1,"Cr");
    printf("=== rgb->ycbcr 420: Y[0]=%u ref=%u | Cb[0]=%u ref=%u | Cr[0]=%u ref=%u ===\n",
           oY[0],rY[0], oCb[0],rCb[0], oCr[0],rCr[0]);
    printf("    (badY=%d badCb=%d badCr=%d)\n", badY,badCb,badCr);
    return report(badY+badCb+badCr, H*W + 2*CH*CW);
}
