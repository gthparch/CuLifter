#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// tma_two_desc(const float* a, const float* b, float* out, int N)
extern "C" void tma_two_desc();

int main() {
    constexpr int N = 256;
    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(i + 1) * 0.3f;
        h_b[i] = (float)(N - i) * 0.2f;
    }
    memset(h_out, 0, sizeof(h_out));

    launchKernel(tma_two_desc, (N + 255) / 256, 256,
                 (float*)h_a, (float*)h_b, (float*)h_out, (int)N);

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
