// host_cpu.cpp — x86 exec-correctness for nvjpeg YCbCr 4:4:4 -> planar RGB.
//
// nvjpeg::ycbcr_to_format_kernel_roi<444(0), false, RGB(3), LaunchParams<32,8>>
//   (nvjpegImage_t in, NppiSize inSize, nvjpegImage_t out, NppiSize roi)
//
// Decoded from full SASS. Identical FFMA conversion math to the 4:2:0 RGB
// variant; the ONLY difference is the chroma index. For subsamp=0 (4:4:4) the
// SASS uses chroma at FULL resolution — there is NO `LEA.HI ; SHF.R 0x1` halving
// (which is present in the 4:2:0 kernel). The chroma column/row use (r,c) directly:
//   Y  = in.ch0[r*pY  + c]            IMAD R0, R7(r), c[0x230](pY), R2(c)
//   Cb = in.ch1[r*pCb + c]            IMAD R0, R7(r), c[0x238](pCb), R2(c)   <- no >>1
//   Cr = in.ch2[r*pCr + c]            IMAD R3, R7(r), c[0x240](pCr), R2(c)   <- no >>1
//   Cbm = (float)Cb - 128 ; Crm = (float)Cr - 128 ; Yf = (float)Y
//   B = fma(Cbm,  1.7719999551773071289, Yf)
//   G = fma(Crm, -0.71413630247116088867, fma(Cbm, -0.34413629770278930664, Yf))
//   R = fma(Crm,  1.4019999504089355469, Yf)
//   each: v = min(max(v,0),255); out = (uint8)truncf(v)   FSEL/FMNMX.NAN/F2I.U32.TRUNC
//   out.ch0[r*q0 + c] = R ; out.ch1[r*q1 + c] = G ; out.ch2[r*q2 + c] = B
// (ROI param @0x298={0,0} => output coords == input coords, all pixels processed.)
//
// const-bank: in.ch0@0x210 ch1@0x218 ch2@0x220, in.pitch0@0x230 p1@0x238 p2@0x240,
//   inSize.w@0x250 h@0x254, out.ch0@0x258 ch1@0x260 ch2@0x268,
//   out.pitch0@0x278 p1@0x280 p2@0x288, roi.x@0x298 roi.y@0x29C.
#include <cstdint>
#include <cstdio>
#include <cmath>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"

extern "C" void
_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t0ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

static uint8_t ct(float v){ v = v>=0?v:0; v = v<255?v:255; return (uint8_t)(uint32_t)truncf(v); }

int main() {
    // 4:4:4 -> chroma at full resolution. Use distinct, discriminating per-pixel
    // values so a wrong chroma index (e.g. accidental >>1) would change outputs.
    constexpr int W = 16, H = 8;
    uint8_t Y[H*W], Cb[H*W], Cr[H*W];
    uint8_t oR[H*W], oG[H*W], oB[H*W];
    uint8_t rR[H*W], rG[H*W], rB[H*W];

    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) {
        Y [r*W+c] = (uint8_t)(r*23 + c*11 + 7);
        Cb[r*W+c] = (uint8_t)(r*31 + c*17 + 40);
        Cr[r*W+c] = (uint8_t)(r*13 + c*29 + 200);
    }
    // independent host reference (bit-exact: fmaf==FFMA, truncf==F2I.TRUNC)
    for (int r = 0; r < H; ++r) for (int c = 0; c < W; ++c) {
        float Yf  = (float)Y [r*W+c];
        float Cbm = (float)Cb[r*W+c] - 128.0f;   // full-res chroma index
        float Crm = (float)Cr[r*W+c] - 128.0f;
        rB[r*W+c] = ct(fmaf(Cbm, 1.7719999551773071289f, Yf));
        rG[r*W+c] = ct(fmaf(Crm,-0.71413630247116088867f, fmaf(Cbm,-0.34413629770278930664f, Yf)));
        rR[r*W+c] = ct(fmaf(Crm, 1.4019999504089355469f, Yf));
    }
    for (int i = 0; i < H*W; ++i) oR[i] = oG[i] = oB[i] = 0xCC;

    launchKernel2D(
        _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t0ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        Dim3{(W+31)/32,(H+7)/8}, Dim3{32,8},
        ArgAt{0x210},(uint8_t*)Y,  ArgAt{0x218},(uint8_t*)Cb, ArgAt{0x220},(uint8_t*)Cr,
        ArgAt{0x230},(uint64_t)W,  ArgAt{0x238},(uint64_t)W,  ArgAt{0x240},(uint64_t)W,
        ArgAt{0x250},(int)W,       ArgAt{0x254},(int)H,
        ArgAt{0x258},(uint8_t*)oR, ArgAt{0x260},(uint8_t*)oG, ArgAt{0x268},(uint8_t*)oB,
        ArgAt{0x278},(uint64_t)W,  ArgAt{0x280},(uint64_t)W,  ArgAt{0x288},(uint64_t)W,
        ArgAt{0x298},(int)0,       ArgAt{0x29C},(int)0);

    int bad = cmp_u8_tol(oR,rR,H*W,1,"R")+cmp_u8_tol(oG,rG,H*W,1,"G")+cmp_u8_tol(oB,rB,H*W,1,"B");
    printf("=== ycbcr->RGB 444 (tol +/-1): px0 RGB=(%u,%u,%u) ref=(%u,%u,%u) ===\n",
           oR[0],oG[0],oB[0], rR[0],rG[0],rB[0]);
    return report(bad, 3*H*W);
}
