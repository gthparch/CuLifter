#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

int main() {
    constexpr int N = 256;
    float h_in[N];
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 17 + 1) * 0.1f;
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    float ref[N];
    fread(ref, sizeof(float), N, f); fclose(f);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i] * 2.0f + (float)(256 & 0xFF);
        if (fabsf(ref[i] - exp) > 1e-3f) { ok = false; break; }
    }
    printf(ok ? "REF VERIFIED\n" : "REF MISMATCH\n");
    return ok ? 0 : 1;
}
