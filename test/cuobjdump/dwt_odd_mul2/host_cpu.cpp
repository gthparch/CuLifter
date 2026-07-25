// Lifted-kernel driver: nvJPEG2000 (closed-source library) sm_80 kernel
//   nvjpeg2k::inverse_dwt_odd_mul2<0>(const int*, size_t, int*, size_t, uint)
// — the odd-length multiply-by-2 step of the inverse discrete wavelet
// transform in the JPEG 2000 decode path.
//
// Provenance: slifter_cufuzz_x86_demo_20260423.tar.gz (hparch_lab share).
// Golden hashes are FNV-1a-64 of the buffers after an n=32 run; the
// April-2026 demo lifter and HEAD f38f870f produce bit-identical buffers
// (verified 2026-06-11). The src-buffer hash guards against stray writes.
//
// Parameter ABI relative to c[0x0][ARG_BASE], read off the SASS:
//   0x00 src (const int*)  0x08 srcStride (size_t)
//   0x10 dst (int*)        0x18 dstStride (size_t)  0x20 n (u32)

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

static constexpr uint64_t GOLDEN_DST = 0x65113f0ed3e20383ULL;
static constexpr uint64_t GOLDEN_SRC = 0x42a3198e69da0383ULL;

enum : uint32_t {
    OFF_SRC        = 0x00,
    OFF_SRC_STRIDE = 0x08,
    OFF_DST        = 0x10,
    OFF_DST_STRIDE = 0x18,
    OFF_N          = 0x20,
};

extern "C" void _ZN8nvjpeg2k20inverse_dwt_odd_mul2ILi0EEEvPKimPimj();

int main() {
    const uint32_t n = 32;
    // Oversize buffers to tolerate the DWT step's stride/alignment assumptions
    // (real library inputs are aligned per-level coefficient planes).
    size_t srcStride = static_cast<size_t>(n) * 4;
    size_t dstStride = static_cast<size_t>(n) * 4;

    std::vector<int32_t> src(srcStride * 64, 1);
    std::vector<int32_t> dst(dstStride * 64, 0);

    writeStructField<const int*>((uint32_t)ARG_BASE + OFF_SRC,        src.data());
    writeStructField<size_t>    ((uint32_t)ARG_BASE + OFF_SRC_STRIDE, srcStride);
    writeStructField<int*>      ((uint32_t)ARG_BASE + OFF_DST,        dst.data());
    writeStructField<size_t>    ((uint32_t)ARG_BASE + OFF_DST_STRIDE, dstStride);
    writeStructField<uint32_t>  ((uint32_t)ARG_BASE + OFF_N,          n);

    constexpr int BLOCK = 128;
    const int gridX = static_cast<int>((n + BLOCK - 1) / BLOCK);

    std::cout << "[driver] launching inverse_dwt_odd_mul2<0> grid=(" << gridX
              << ") block=(" << BLOCK << ") n=" << n << "\n";

    launchKernel(_ZN8nvjpeg2k20inverse_dwt_odd_mul2ILi0EEEvPKimPimj, gridX, BLOCK);

    uint64_t hDst = fnv(reinterpret_cast<const uint8_t*>(dst.data()), dst.size() * 4);
    uint64_t hSrc = fnv(reinterpret_cast<const uint8_t*>(src.data()), src.size() * 4);
    std::cout << "[driver] dst=" << std::hex << hDst << " src=" << hSrc << std::dec << "\n";

    if (hDst == GOLDEN_DST && hSrc == GOLDEN_SRC) {
        std::cout << "TEST PASSED\n";
        return 0;
    }
    std::cout << "TEST FAILED: hash mismatch (want dst=" << std::hex << GOLDEN_DST
              << " src=" << GOLDEN_SRC << std::dec << ")\n";
    return 1;
}
