#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA5b_bfe_bfi();

int main(int argc, char** argv) {
    const int N = 64;
    unsigned int h_in[64], h_out[64];
    for (int i = 0; i < N; i++) h_in[i] = 0x12345678u + i * 0x01010101u;

    launchKernel(ISA5b_bfe_bfi, 1, 64, (void*)h_out, (const void*)h_in, N);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(unsigned), N, f);
        fclose(f);
    }

    int pass = 1;
    for (int i = 0; i < N; i++) {
        unsigned int byte1 = (h_in[i] >> 8) & 0xFF;
        unsigned int expected = byte1 << 16;
        if (h_out[i] != expected) {
            printf("FAIL [%d]: got 0x%08x, expected 0x%08x\n", i, h_out[i], expected);
            pass = 0;
        }
    }

    printf(pass ? "TEST PASSED\n" : "TEST FAILED\n");
    return pass ? 0 : 1;
}
