#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

int main() {
    constexpr int GRID = 4;
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    float ref[GRID];
    fread(ref, sizeof(float), GRID, f); fclose(f);
    bool ok = true;
    for (int i = 0; i < GRID; i++)
        if (fabsf(ref[i] - 256.0f) > 1e-3f) { ok = false; break; }
    printf(ok ? "REF VERIFIED\n" : "REF MISMATCH\n");
    return ok ? 0 : 1;
}
