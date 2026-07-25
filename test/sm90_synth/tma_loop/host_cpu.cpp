#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// tma_loop(const float* in, float* out, int N, int K)
extern "C" void tma_loop();

int main() {
    constexpr int N = 64;
    constexpr int K = 4;
    constexpr int IN_SIZE = N * K;
    float h_in[IN_SIZE], h_out[N];
    for (int i = 0; i < IN_SIZE; i++) h_in[i] = (float)(i + 1) * 0.01f;
    memset(h_out, 0, sizeof(h_out));

    launchKernel(tma_loop, (N + 255) / 256, 256,
                 (float*)h_in, (float*)h_out, (int)N, (int)K);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    float ref[N];
    fread(ref, sizeof(float), N, f); fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - ref[i]) > 1e-3f) {
            printf("FAIL at %d: got %f ref %f\n", i, h_out[i], ref[i]);
            ok = false; break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "FAIL\n");
    return ok ? 0 : 1;
}
