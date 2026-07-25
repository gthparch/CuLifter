// host_cpu.cpp for sm90_wgmma_rs — x86 execution-correctness driver for a
// genuine sm_90a HGMMA (SS-form descriptor) tensor-core kernel.
//
// The lifted kernel `fa_wgmma()` reads gA (64x16 fp16), gB (16x16 fp16) from
// global memory via desc[UR8] addressing, stages them into shared memory (STS),
// runs HGMMA.64x16x16.F32 (decoded by the x86 lifter into a real A*B product
// read back out of shared memory), and stores D (64x16 fp32) to gD.
//
// The caller supplies the checked-in A/B inputs and native-H200 reference
// output. We run the lifted kernel across all 128 warpgroup threads and compare
// D against that reference. Tensor-core HGMMA decomposed to scalar fp32 FMA is
// not bit-exact vs the hardware path, so we assert a documented fp tolerance.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void fa_wgmma();

template <typename T>
static int load_bin(const char* path, T* dst, size_t n) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("FAIL: cannot open %s\n", path); return 1; }
    size_t got = fread(dst, sizeof(T), n, f);
    fclose(f);
    if (got != n) { printf("FAIL: short read %s (%zu/%zu)\n", path, got, n); return 1; }
    return 0;
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (frac == 0) { bits = sign << 31; }
        else {
            // subnormal
            int e = -14;
            while (!(frac & 0x400)) { frac <<= 1; e--; }
            frac &= 0x3FF;
            bits = (sign << 31) | ((uint32_t)(e + 127) << 23) | (frac << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    float out; memcpy(&out, &bits, 4); return out;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s DATA_DIR\n", argv[0]);
        return 2;
    }
    const char* DATA = argv[1];
    constexpr int M = 64, K = 16, N = 16;
    char p[512];

    // half is stored as raw uint16 in the bins.
    uint16_t hA[M * K], hB[N * K];
    float refD[M * N], outD[M * N];
    snprintf(p, sizeof(p), "%s/in_A.bin", DATA);  if (load_bin(p, hA, M * K)) return 1;
    snprintf(p, sizeof(p), "%s/in_B.bin", DATA);  if (load_bin(p, hB, N * K)) return 1;
    snprintf(p, sizeof(p), "%s/ref_D.bin", DATA); if (load_bin(p, refD, M * N)) return 1;

    // Device-side buffers the lifted kernel reads/writes (plain host memory; the
    // x86 wrapper runs on the CPU so global pointers are ordinary addresses).
    // gA/gB are fp16 -> we hand the kernel the raw uint16 array (the lifted
    // LDG.E.U16 reads exactly these bytes and converts).
    memset(outD, 0, sizeof(outD));

    launchKernel(fa_wgmma, 1, 128,
                 ArgAt{0x210}, (uint16_t*)hA,
                 ArgAt{0x218}, (uint16_t*)hB,
                 ArgAt{0x220}, (float*)outD);

    // Compare against native H200 ground truth.
    double maxabs = 0.0, maxrel = 0.0;
    int bad = 0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs((double)outD[i] - (double)refD[i]);
        double denom = fabs((double)refD[i]);
        double rel = denom > 1e-6 ? diff / denom : diff;
        if (diff > maxabs) maxabs = diff;
        if (rel > maxrel)  maxrel = rel;
    }
    // Tolerance: scalar-decomposed HGMMA accumulates in fp32 in a different
    // order than the HW tensor core (and inputs are fp16), so a small abs
    // tolerance is expected. The values are O(0.1); 1e-2 abs is a tight bound.
    const double EPS_ABS = 1e-2;

    printf("=== x86-lifted HGMMA D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", outD[n]); printf("\n");
    printf("=== native H200   D row0 ===\n");
    for (int n = 0; n < N; n++) printf("%.4f ", refD[n]); printf("\n");
    printf("max |x86 - H200| abs = %.6f  rel = %.6f  (eps_abs=%.0e)\n",
           maxabs, maxrel, EPS_ABS);

    for (int i = 0; i < M * N; i++)
        if (fabs((double)outD[i] - (double)refD[i]) > EPS_ABS) bad++;

    if (bad == 0) { printf("TEST PASSED\n"); return 0; }
    printf("FAIL: %d / %d elements exceed eps_abs\n", bad, M * N);
    return 1;
}
