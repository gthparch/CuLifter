// host_cpu.cpp — x86 exec-correctness for the dominant nvjpeg decode kernel.
//
// nvjpeg::ycbcr_to_format_kernel_roi<420, false, BGR(4), LaunchParams<32,8>>
//   (nvjpegImage_t in, NppiSize inSize, nvjpegImage_t out, NppiSize roi)
//
// Decoded from full SASS. Planar YCbCr 4:2:0 -> planar BGR. Per output pixel (r,c):
//   Y  = in.ch0[r*pY + c]
//   Cb = in.ch1[(r>>1)*pCb + (c>>1)]      (4:2:0 -> chroma at half res)
//   Cr = in.ch2[(r>>1)*pCr + (c>>1)]
//   Cbm = (float)Cb - 128 ; Crm = (float)Cr - 128 ; Yf = (float)Y
//   B = fma(Cbm, 1.7719999551773071289, Yf)                         <- FFMA
//   G = fma(Crm,-0.71413630247116088867, fma(Cbm,-0.34413629770278930664, Yf))
//   R = fma(Crm, 1.4019999504089355469, Yf)
//   each: v = min(max(v,0),255); out = (uint8)truncf(v)             <- FSEL/FMNMX/F2I.TRUNC
//   out.ch0[r*q0 + c] = B ; out.ch1[r*q1 + c] = G ; out.ch2[r*q2 + c] = R
// (ROI param @0x298={0,0} => output coords == input coords, all pixels processed.)
//
// const-bank: in.ch0@0x210 ch1@0x218 ch2@0x220, in.pitch0@0x230 p1@0x238 p2@0x240,
//   inSize.w@0x250 h@0x254, out.ch0@0x258 ch1@0x260 ch2@0x268,
//   out.pitch0@0x278 p1@0x280 p2@0x288, roi.x@0x298 roi.y@0x29C.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void
_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t2ELb0EL20nvjpegOutputFormat_t4ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

static uint8_t clamp_trunc(float v) {
    v = v >= 0.0f ? v : 0.0f;
    v = v < 255.0f ? v : 255.0f;          // min(v,255)
    return (uint8_t)(uint32_t)truncf(v);  // F2I.U32.TRUNC
}

int main() {
    constexpr int W = 16, H = 8;          // even dims (4:2:0)
    constexpr int CW = W / 2, CH = H / 2;
    uint8_t Y[H*W], Cb[CH*CW], Cr[CH*CW];
    uint8_t oB[H*W], oG[H*W], oR[H*W];
    uint8_t rB[H*W], rG[H*W], rR[H*W];

    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c)
        Y[r*W+c] = (uint8_t)(r*23 + c*11 + 7);
    for (int r = 0; r < CH; ++r) for (int c = 0; c < CW; ++c) {
        Cb[r*CW+c] = (uint8_t)(r*31 + c*17 + 40);
        Cr[r*CW+c] = (uint8_t)(r*13 + c*29 + 200);
    }
    // host reference (bit-exact: fmaf == FFMA, truncf == F2I.TRUNC)
    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) {
        float Yf  = (float)Y[r*W+c];
        float Cbm = (float)Cb[(r>>1)*CW + (c>>1)] - 128.0f;
        float Crm = (float)Cr[(r>>1)*CW + (c>>1)] - 128.0f;
        rB[r*W+c] = clamp_trunc(fmaf(Cbm, 1.7719999551773071289f, Yf));
        rG[r*W+c] = clamp_trunc(fmaf(Crm,-0.71413630247116088867f,
                                fmaf(Cbm,-0.34413629770278930664f, Yf)));
        rR[r*W+c] = clamp_trunc(fmaf(Crm, 1.4019999504089355469f, Yf));
    }
    for (int i=0;i<H*W;++i){ oB[i]=oG[i]=oR[i]=0xCC; }

    launchKernel2D(
        _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t2ELb0EL20nvjpegOutputFormat_t4ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        Dim3{(W+31)/32,(H+7)/8}, Dim3{32,8},
        ArgAt{0x210}, (uint8_t*)Y,  ArgAt{0x218}, (uint8_t*)Cb, ArgAt{0x220}, (uint8_t*)Cr,
        ArgAt{0x230}, (uint64_t)W,  ArgAt{0x238}, (uint64_t)CW, ArgAt{0x240}, (uint64_t)CW,
        ArgAt{0x250}, (int)W,       ArgAt{0x254}, (int)H,
        ArgAt{0x258}, (uint8_t*)oB, ArgAt{0x260}, (uint8_t*)oG, ArgAt{0x268}, (uint8_t*)oR,
        ArgAt{0x278}, (uint64_t)W,  ArgAt{0x280}, (uint64_t)W,  ArgAt{0x288}, (uint64_t)W,
        ArgAt{0x298}, (int)0,       ArgAt{0x29C}, (int)0);

    int bad = 0;
    for (int i = 0; i < H*W; ++i) {
        if (oB[i]!=rB[i]||oG[i]!=rG[i]||oR[i]!=rR[i]) {
            if (bad<8) printf("  px %d: got BGR(%u,%u,%u) want (%u,%u,%u)\n",
                              i,oB[i],oG[i],oR[i],rB[i],rG[i],rR[i]);
            ++bad;
        }
    }
    printf("=== ycbcr->BGR 420: px0 BGR=(%u,%u,%u) ref=(%u,%u,%u) ===\n",
           oB[0],oG[0],oR[0],rB[0],rG[0],rR[0]);
    printf("mismatches = %d / %d\n", bad, H*W);
    if (bad==0){ printf("TEST PASSED\n"); return 0; }
    printf("FAIL\n"); return 1;
}
