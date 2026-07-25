#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdint>
#include "kernel_wrapper.h"
// Real x86 host: runs the LIFTED kernel (3-vector normalize: MUFU.RSQ / div,
// FMUL) and compares its output against the captured GPU reference. Inputs +
// launch config mirror ref_host.cu; ref layout is ox[256], oy[256], oz[256].
extern "C" void DM_normalize();

int main() {
    constexpr int N = 256;
    float hx[N], hy[N], hz[N], ox[N], oy[N], oz[N];
    for (int i = 0; i < N; i++) {
        hx[i] = (float)((i * 3 + 1) % 17) * 0.3f;
        hy[i] = (float)((i * 5 + 2) % 13) * 0.4f;
        hz[i] = (float)((i * 7 + 3) % 11) * 0.5f;
    }
    memset(ox, 0, sizeof(ox)); memset(oy, 0, sizeof(oy)); memset(oz, 0, sizeof(oz));

    launchKernel(DM_normalize, 1, 256, hx, hy, hz, ox, oy, oz, N);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("No ref\n"); return 1; }
    float ref[N * 3];
    if (fread(ref, 4, N * 3, f) != (size_t)(N * 3)) { printf("Short ref\n"); fclose(f); return 1; }
    fclose(f);

    // MUFU.RSQ on the SFU differs from the host's 1/sqrtf by a few ULP, so use a
    // small relative tolerance — a real lift bug shows up as a large mismatch.
    bool ok = true;
    const float* got[3]  = { ox, oy, oz };
    for (int c = 0; c < 3 && ok; c++)
        for (int i = 0; i < N; i++)
            if (fabsf(got[c][i] - ref[c * N + i]) > 1e-3f) {
                printf("FAIL comp %d idx %d: got %f exp %f\n", c, i, got[c][i], ref[c * N + i]);
                ok = false; break;
            }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
