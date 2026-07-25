// nvjpeg::ycbcr_to_format_kernel_roi<GRAY(6), false, RGB(3), LaunchParams<32,8>>
// Grayscale path: loads only Y, writes Y to all 3 output planes (R=G=B=Y).
// Pure data movement -> exact (tol=0). Demonstrates the same kernel template as
// the 420->BGR variant but the GRAY subsampling specialization.
// const-bank: in.ch0@0x210 pitch@0x230, inSize@0x250, out.ch0@0x258 ch1@0x260
//   ch2@0x268, out.pitch@0x278/80/88, roi@0x298 (=0).
#include <cstdint>
#include <cstdio>
#include "kernel_wrapper.h"
#include "nvlib_compare.h"

extern "C" void
_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t6ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

int main() {
    constexpr int W = 16, H = 8;
    uint8_t Y[H*W], o0[H*W], o1[H*W], o2[H*W];
    for (int i=0;i<H*W;++i){ Y[i]=(uint8_t)(i*37+5); o0[i]=o1[i]=o2[i]=0xCC; }

    launchKernel2D(
        _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t6ELb0EL20nvjpegOutputFormat_t3ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        Dim3{(W+31)/32,(H+7)/8}, Dim3{32,8},
        ArgAt{0x210}, (uint8_t*)Y, ArgAt{0x230}, (uint64_t)W,
        ArgAt{0x250}, (int)W,     ArgAt{0x254}, (int)H,
        ArgAt{0x258}, (uint8_t*)o0, ArgAt{0x260}, (uint8_t*)o1, ArgAt{0x268}, (uint8_t*)o2,
        ArgAt{0x278}, (uint64_t)W, ArgAt{0x280}, (uint64_t)W, ArgAt{0x288}, (uint64_t)W,
        ArgAt{0x298}, (int)0,      ArgAt{0x29C}, (int)0);

    int bad = cmp_u8_tol(o0,Y,H*W,0,"R") + cmp_u8_tol(o1,Y,H*W,0,"G") + cmp_u8_tol(o2,Y,H*W,0,"B");
    printf("=== ycbcr GRAY->RGB: o0[0..3]=%u %u %u %u  Y=%u %u %u %u ===\n",
           o0[0],o0[1],o0[2],o0[3], Y[0],Y[1],Y[2],Y[3]);
    return report(bad, 3*H*W);
}
