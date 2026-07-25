// Scalar F2F.BF16.F32 (non-PACK) test: cvt.rn.bf16.f32 + cvt.rz.bf16.f32 in
// the same kernel cannot pair into F2FP.BF16.F32.PACK_AB, so ptxas emits the
// scalar forms F2F.BF16.F32 and F2F.BF16.F32.RZ — the cuBLAS-internal shape
// (307 sites in the cublas134 sm_90 corpus). The bf16 bits land in the dest
// reg's LOW half (corpus consumers read .H0 / SHF.L 0x10 back to f32).
// Also exercises the widening direction cvt.f32.bf16 (bits << 16, exact).

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>

extern "C" __global__ void kern_f2f_bf16_scalar(const float* __restrict__ in,
                                                uint16_t* __restrict__ out_rn,
                                                uint16_t* __restrict__ out_rz,
                                                float* __restrict__ out_widen,
                                                int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = in[i];
        uint16_t brn, brz;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(brn) : "f"(v));
        asm("cvt.rz.bf16.f32 %0, %1;" : "=h"(brz) : "f"(v));
        float w;
        asm("cvt.f32.bf16 %0, %1;" : "=f"(w) : "h"(brn));
        out_rn[i] = brn;
        out_rz[i] = brz;
        out_widen[i] = w;
    }
}

// Exact host conversions (IEEE RN-even / RZ; bf16 = f32's top 16 bits).
// Native-H100 validation pinned cvt.rn.bf16.f32(0x7FC00000) == 0x7FFF:
// sign and payload are dropped.
static uint16_t f32_to_bf16_rn(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if (std::isnan(f)) return (uint16_t)0x7FFF;
    uint32_t lsb = (u >> 16) & 1;
    return (uint16_t)((u + 0x7FFF + lsb) >> 16);
}
static uint16_t f32_to_bf16_rz(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if (std::isnan(f)) return (uint16_t)0x7FFF;
    return (uint16_t)(u >> 16);
}

static void fill_inputs(float* h_in, int n) {
    // Exponent sweep, signs, RN-even ties (low half exactly 0x8000 with
    // even/odd bf16 lsb), values just above/below ties, denormals, ±Inf,
    // NaN variants (quiet/signaling, signed, payloads — all -> 0x7FFF).
    uint32_t pats[] = {
        0x3F800000u, 0xBF800000u,             // +-1.0
        0x3F808000u, 0x3F818000u,             // RN ties: even keeps, odd bumps
        0x3F807FFFu, 0x3F818001u,             // just below / above tie
        0x7F7F8000u, 0x7F7FFFFFu,             // near f32 max (tie -> bf16 inf)
        0x00008000u, 0x00018000u, 0x807FFFFFu, // denormals, signed
        0x7F800000u, 0xFF800000u,             // +-Inf
        0x7FC00000u, 0xFFC00001u,             // quiet NaNs, signed + payload
        0x7F800001u, 0xFF80FFFFu,             // signaling NaNs, signed
        0x00000000u, 0x80000000u,             // +-0
    };
    int npat = sizeof(pats) / sizeof(pats[0]);
    for (int i = 0; i < n; i++) {
        uint32_t u;
        if (i < npat) u = pats[i];
        else u = (0x30000000u + 0x01010101u * (uint32_t)i) ^ ((uint32_t)i << 13);
        memcpy(&h_in[i], &u, 4);
    }
}

int main() {
    constexpr int N = 128;
    constexpr int blockDim_x = 128;
    constexpr int gridDim_x = 1;

    float h_in[N], h_widen[N];
    uint16_t h_rn[N], h_rz[N];
    fill_inputs(h_in, N);

    float *d_in, *d_widen;
    uint16_t *d_rn, *d_rz;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_rn, N * sizeof(uint16_t));
    cudaMalloc(&d_rz, N * sizeof(uint16_t));
    cudaMalloc(&d_widen, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    kern_f2f_bf16_scalar<<<gridDim_x, blockDim_x>>>(d_in, d_rn, d_rz, d_widen, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_rn, d_rn, N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rz, d_rz, N * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_widen, d_widen, N * sizeof(float), cudaMemcpyDeviceToHost);

    FILE* f = fopen("ref_output.bin", "wb");
    fwrite(h_rn, sizeof(uint16_t), N, f);
    fwrite(h_rz, sizeof(uint16_t), N, f);
    fwrite(h_widen, sizeof(float), N, f);
    fclose(f);

    int n_fail = 0;
    for (int i = 0; i < N; i++) {
        uint16_t ern = f32_to_bf16_rn(h_in[i]);
        uint16_t erz = f32_to_bf16_rz(h_in[i]);
        uint32_t ew32 = (uint32_t)ern << 16;
        uint32_t gw32; memcpy(&gw32, &h_widen[i], 4);
        if (h_rn[i] != ern) n_fail++;
        if (h_rz[i] != erz) n_fail++;
        if (gw32 != ew32) n_fail++;
    }

    printf(n_fail == 0 ? "TEST PASSED\n" : "TEST FAILED (%d mismatches)\n", n_fail);

    cudaFree(d_in); cudaFree(d_rn); cudaFree(d_rz); cudaFree(d_widen);
    return n_fail == 0 ? 0 : 1;
}
