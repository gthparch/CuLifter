#include <cstdio>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B_ldl64();

int main() {
    constexpr int N = 128;
    double h_in[N], h_out[N];
    memset(h_out, 0, N * sizeof(double));
    for (int i = 0; i < N; i++) h_in[i] = (double)(i + 1) * 0.01;

    launchKernel(B_ldl64, 1, 128, h_in, h_out, N);

    // Load GPU reference
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) {
        // No ref — just smoke test
        bool ok = !std::isnan(h_out[0]) && h_out[0] != 0.0;
        printf(ok ? "TEST PASSED\n" : "FAIL: output is zero or NaN\n");
        return ok ? 0 : 1;
    }
    double ref[N];
    fread(ref, sizeof(double), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (std::isnan(h_out[i]) || std::fabs(h_out[i] - ref[i]) > 1e-6 * std::fabs(ref[i]) + 1e-12) {
            printf("FAIL at %d: got %g ref %g\n", i, h_out[i], ref[i]);
            ok = false;
            break;
        }
    }
    if (ok) printf("TEST PASSED\n");
    return ok ? 0 : 1;
}
