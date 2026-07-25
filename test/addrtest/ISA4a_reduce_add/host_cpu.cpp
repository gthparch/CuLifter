#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA4a_reduce_add();

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[1] = {0};
    for (int i = 0; i < N; i++) h_in[i] = i + 1;

    // The x86 wrapper runs all 32 warp lanes concurrently, so __reduce_add_sync
    // is a REAL warp-wide reduction (matching the GPU): every lane receives the
    // sum of all members, and tid==0 stores it. Expected = sum over the warp
    // (1+2+...+32 = 528), not the old single-lane identity (in[0]=1).
    int expected = 0;
    for (int i = 0; i < N; i++) expected += h_in[i];

    launchKernel(ISA4a_reduce_add, 1, 32, (void*)h_out, (const void*)h_in, N);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    if (h_out[0] == expected) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected %d\n", h_out[0], expected);
        return 1;
    }
    return 0;
}
