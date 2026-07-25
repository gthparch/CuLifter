// Lifted-kernel driver: nvJPEG (closed-source library) sm_80 kernel
//   nvjpeg::ycbcr_to_format_kernel_roi<NVJPEG_CSS_GRAY, false, OF=5,
//                                      LaunchParams<32, 8>>
//       (nvjpegImage_t in, NppiSize inROI, nvjpegImage_t out, NppiSize outROI)
//
// Provenance: slifter_cufuzz_x86_demo_20260423.tar.gz (hparch_lab share).
// Golden hashes are FNV-1a-64 of the buffers after a 480x320 run; the
// April-2026 demo lifter and HEAD f38f870f produce bit-identical buffers
// (verified 2026-06-11). The input-buffer hash guards against stray writes.
//
// Parameter ABI relative to c[0x0][ARG_BASE], read off the SASS — the kernel
// only touches channel[0]/pitch[0] of each nvjpegImage_t plus both NppiSize:
//   0x00 in.channel[0] (u8*)   0x20 in.pitch[0] (size_t)   0x40 inROI (2xi32)
//   0x48 out.channel[0] (u8*)  0x68 out.pitch[0] (size_t)  0x88 outROI

#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "kernel_wrapper.h"

static uint64_t fnv(const uint8_t* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static constexpr uint64_t GOLDEN_OUT = 0x1a5bcfc145f96383ULL;
static constexpr uint64_t GOLDEN_IN  = 0xb795fb241b112383ULL;

static constexpr int NVJPEG_MAX_COMPONENT = 4;
struct nvjpegImage_t {
    unsigned char* channel[NVJPEG_MAX_COMPONENT];
    size_t         pitch[NVJPEG_MAX_COMPONENT];
};
static_assert(sizeof(nvjpegImage_t) == 64);

struct NppiSize { int32_t width, height; };
static_assert(sizeof(NppiSize) == 8);

enum : uint32_t {
    OFF_IN_CHANNEL0  = 0x00,
    OFF_IN_PITCH0    = 0x20,
    OFF_IN_ROI       = 0x40,
    OFF_OUT_CHANNEL0 = 0x48,
    OFF_OUT_PITCH0   = 0x68,
    OFF_OUT_ROI      = 0x88,
};

extern "C" void
_ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t6ELb0EL20nvjpegOutputFormat_t5ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_();

int main() {
    const int width = 480, height = 320;
    size_t pitch = static_cast<size_t>(width);

    std::vector<uint8_t> inBuf (pitch * height, 0x80);   // grey plane
    std::vector<uint8_t> outBuf(pitch * height * 3, 0);  // RGB output

    nvjpegImage_t inImg{}, outImg{};
    inImg.channel[0]  = inBuf.data();   inImg.pitch[0]  = pitch;
    outImg.channel[0] = outBuf.data();  outImg.pitch[0] = pitch * 3;

    NppiSize inROI { width, height }, outROI{ width, height };

    writeStructField<unsigned char*>((uint32_t)ARG_BASE + OFF_IN_CHANNEL0,  inImg.channel[0]);
    writeStructField<size_t>        ((uint32_t)ARG_BASE + OFF_IN_PITCH0,    inImg.pitch[0]);
    writeStructField<NppiSize>      ((uint32_t)ARG_BASE + OFF_IN_ROI,       inROI);
    writeStructField<unsigned char*>((uint32_t)ARG_BASE + OFF_OUT_CHANNEL0, outImg.channel[0]);
    writeStructField<size_t>        ((uint32_t)ARG_BASE + OFF_OUT_PITCH0,   outImg.pitch[0]);
    writeStructField<NppiSize>      ((uint32_t)ARG_BASE + OFF_OUT_ROI,      outROI);

    // LaunchParams<32, 8> tile; blocks cover the image (library launch shape).
    constexpr int BX = 32, BY = 8;
    Dim3 grid  { (width + BX - 1) / BX, (height + BY - 1) / BY, 1 };
    Dim3 block { BX, BY, 1 };

    std::cout << "[driver] launching ycbcr_to_format_kernel_roi<OF=5> grid=("
              << grid.x << "," << grid.y << ") block=("
              << block.x << "," << block.y << ")\n";

    launchKernel2D(
        _ZN6nvjpeg26ycbcr_to_format_kernel_roiIL25nvjpegChromaSubsampling_t6ELb0EL20nvjpegOutputFormat_t5ENS_26ConvertToFormatDispatchROI12LaunchParamsILi32ELi8EEEEEv13nvjpegImage_t8NppiSizeS6_S7_,
        grid, block);

    uint64_t hOut = fnv(outBuf.data(), outBuf.size());
    uint64_t hIn  = fnv(inBuf.data(),  inBuf.size());
    std::cout << "[driver] out=" << std::hex << hOut << " in=" << hIn << std::dec << "\n";

    if (hOut == GOLDEN_OUT && hIn == GOLDEN_IN) {
        std::cout << "TEST PASSED\n";
        return 0;
    }
    std::cout << "TEST FAILED: hash mismatch (want out=" << std::hex << GOLDEN_OUT
              << " in=" << GOLDEN_IN << std::dec << ")\n";
    return 1;
}
