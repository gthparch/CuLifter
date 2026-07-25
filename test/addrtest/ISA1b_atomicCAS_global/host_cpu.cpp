#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA1b_atomicCAS_global();

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[1] = {0};
    for (int i = 0; i < N; i++) h_in[i] = 100 + i;

    launchKernel(ISA1b_atomicCAS_global, 1, 32, (void*)h_out, (const void*)h_in, N);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    // x86 single-thread: tid=0 always wins, sets out[0] = 100
    if (h_out[0] == 100) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected 100\n", h_out[0]);
        return 1;
    }
    return 0;
}
