#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// tma_desc_scalar(const float* in, float* out, int N, float alpha, float beta)
extern "C" void tma_desc_scalar();

int main() {
    constexpr int N = 256;
    float alpha = 2.5f, beta = 1.0f;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.1f;
    memset(h_out, 0, sizeof(h_out));

    launchKernel(tma_desc_scalar, (N + 255) / 256, 256,
                 (float*)h_in, (float*)h_out, (int)N, alpha, beta);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    float ref[N];
    fread(ref, sizeof(float), N, f); fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - ref[i]) > 1e-5f) {
            printf("FAIL at %d: got %f ref %f\n", i, h_out[i], ref[i]);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    return ok ? 0 : 1;
}
