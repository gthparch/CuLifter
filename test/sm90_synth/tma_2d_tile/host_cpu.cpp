#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// tma_2d_tile(const float* in, float* out, int rows, int cols)
extern "C" void tma_2d_tile();

int main() {
    constexpr int ROWS = 16, COLS = 16;
    constexpr int N = ROWS * COLS;
    float h_in[N], h_out[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.1f;
    memset(h_out, 0, sizeof(h_out));

    launchKernel2D(tma_2d_tile,
                   Dim3{(COLS + 15) / 16, (ROWS + 15) / 16},
                   Dim3{16, 16},
                   (float*)h_in, (float*)h_out, (int)ROWS, (int)COLS);

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
