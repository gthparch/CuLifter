#include <cstdio>
#include <cstring>
#include <cmath>
#include "kernel_wrapper.h"

extern "C" void kern_curand_xorwow();

int main() {
    constexpr int N = 64;
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));

    launchKernel(kern_curand_xorwow, 1, 64, h_out, N);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("no ref\n"); return 1; }
    float ref[N];
    fread(ref, 4, N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (std::isnan(h_out[i]) || std::fabs(h_out[i] - ref[i]) > 1e-7f) {
            printf("FAIL at %d: got %.9g ref %.9g\n", i, h_out[i], ref[i]);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
