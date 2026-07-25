#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "kernel_wrapper.h"
// Real x86 host: runs the LIFTED WMMA kernel (HMMA tensor-core emulation) and
// compares the 16x16 result against the captured GPU reference. half inputs are
// built with the wrapper's f2h_bits (identical bit pattern to __float2half), so
// both sides start from the same fp16 values. Mirrors ref_host.cu.
extern "C" void TC0_hmma_basic();

int main() {
    constexpr int M = 16, N = 16, K = 16;
    uint16_t h_A[M * K], h_B[K * N];
    float h_C[M * N];
    for (int i = 0; i < M * K; i++) h_A[i] = f2h_bits((float)((i % 5 + 1) * 0.1f));
    for (int i = 0; i < K * N; i++) h_B[i] = f2h_bits((float)((i % 7 + 1) * 0.1f));
    memset(h_C, 0, sizeof(h_C));

    launchKernel(TC0_hmma_basic, 1, 32, h_A, h_B, h_C, K);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("No ref\n"); return 1; }
    float ref[M * N];
    if (fread(ref, 4, M * N, f) != (size_t)(M * N)) { printf("Short ref\n"); fclose(f); return 1; }
    fclose(f);

    // fp16 inputs accumulated in fp32: GPU tensor core vs lifted scalar
    // FMUL/FADD differ only by accumulation order over 16 terms (tiny). A real
    // HMMA lift bug shows up as a large mismatch / NaN.
    bool ok = true;
    for (int i = 0; i < M * N; i++)
        if (std::isnan(h_C[i]) || fabsf(h_C[i] - ref[i]) > 1e-2f) {
            printf("FAIL %d: got %f exp %f\n", i, h_C[i], ref[i]);
            ok = false; break;
        }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
