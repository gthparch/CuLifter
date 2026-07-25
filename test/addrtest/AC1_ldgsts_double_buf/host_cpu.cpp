#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "kernel_wrapper.h"
// Real x86 host: runs the LIFTED kernel (exercising the LDGSTS / cp.async ->
// shared path) and compares its output against the captured GPU reference
// (ref_output.bin). Inputs + launch config mirror ref_host.cu exactly.
extern "C" void AC1_ldgsts_double_buf();

int main() {
    constexpr int N = 512;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) { h_in[i] = (float)(i + 1); h_out[i] = 0.0f; }

    launchKernel(AC1_ldgsts_double_buf, 1, 256, h_in, h_out, N);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("No ref\n"); return 1; }
    float ref[N];
    if (fread(ref, 4, N, f) != (size_t)N) { printf("Short ref\n"); fclose(f); return 1; }
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - ref[i]) > 1e-3f) {
            printf("FAIL %d: got %f exp %f\n", i, h_out[i], ref[i]);
            ok = false; break;
        }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
