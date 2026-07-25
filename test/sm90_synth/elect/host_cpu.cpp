#include <cstdio>
#include <cstdint>
#include <cstring>
#include "kernel_wrapper.h"

int main() {
    constexpr int N = 256;
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { printf("Cannot open ref_output.bin\n"); return 1; }
    int ref[N];
    fread(ref, sizeof(int), N, f); fclose(f);
    // Lane 0 of each warp should be elected
    bool ok = (ref[0] == 1);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
