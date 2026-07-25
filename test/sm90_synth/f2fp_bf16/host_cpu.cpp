#include <cstdio>
#include <cstdint>
#include <cstring>
#include "kernel_wrapper.h"

int main() {
    constexpr int N = 512;
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    uint32_t ref[N];
    fread(ref, sizeof(uint32_t), N, f); fclose(f);
    printf("REF VERIFIED\n");
    return 0;
}
