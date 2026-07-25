#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

int main() {
    constexpr int R = 32, C = 32;
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    float ref[C * R];
    fread(ref, sizeof(float), C * R, f); fclose(f);
    // Check transpose: ref[j*R+i] should be (i*C+j+1)
    bool ok = true;
    for (int i = 0; i < R && ok; i++)
        for (int j = 0; j < C && ok; j++)
            if (fabsf(ref[j * R + i] - (float)(i * C + j + 1)) > 1e-5f) ok = false;
    printf(ok ? "REF VERIFIED\n" : "REF MISMATCH\n");
    return ok ? 0 : 1;
}
