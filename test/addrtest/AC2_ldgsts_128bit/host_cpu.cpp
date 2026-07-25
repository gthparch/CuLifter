#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "kernel_wrapper.h"
// Real x86 host: runs the LIFTED kernel (128-bit float4 LDGSTS / cp.async ->
// shared) and compares its output against the captured GPU reference. Inputs +
// launch config mirror ref_host.cu (float4 in[i]={4i+1,4i+2,4i+3,4i+4}).
extern "C" void AC2_ldgsts_128bit();

int main() {
    constexpr int N = 64;            // float4 count
    constexpr int F = N * 4;         // scalar float count
    float h_in[F], h_out[F];
    for (int i = 0; i < F; i++) { h_in[i] = (float)(i + 1); h_out[i] = 0.0f; }

    launchKernel(AC2_ldgsts_128bit, 1, 64, h_in, h_out, N);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("No ref\n"); return 1; }
    float ref[F];
    if (fread(ref, 4, F, f) != (size_t)F) { printf("Short ref\n"); fclose(f); return 1; }
    fclose(f);

    bool ok = true;
    for (int i = 0; i < F; i++)
        if (fabsf(h_out[i] - ref[i]) > 1e-3f) {
            printf("FAIL %d: got %f exp %f\n", i, h_out[i], ref[i]);
            ok = false; break;
        }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
