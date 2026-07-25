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
    // dp4a(i=0): (-3)*(-6) + (-2)*(-4) + (-1)*(-3) + (-5)*(-2) = 18+8+3+10 = 39
    bool ok = (ref[0] == 39);
    printf(ok ? "TEST PASSED\n" : "TEST FAILED (ref[0]=%d, expected 39)\n", ref[0]);
    return ok ? 0 : 1;
}
