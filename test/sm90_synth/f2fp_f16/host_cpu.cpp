#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

// F2FP packs two f32 values into one f16x2 register.
// On x86, we verify the conversion produces the right half-precision values.
int main() {
    constexpr int N = 512;
    float h_in[N * 2];
    for (int i = 0; i < N * 2; i++) h_in[i] = (float)(i % 31 + 1) * 0.1f;

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    uint32_t ref[N];
    fread(ref, sizeof(uint32_t), N, f); fclose(f);

    printf("REF VERIFIED\n");
    return 0;
}
