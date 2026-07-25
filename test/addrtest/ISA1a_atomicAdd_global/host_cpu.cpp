#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA1a_atomicAdd_global();

int main(int argc, char** argv) {
    const int N = 256;
    int h_in[256], h_out[1] = {0};
    int expected = 0;
    for (int i = 0; i < N; i++) { h_in[i] = i + 1; expected += h_in[i]; }

    launchKernel(ISA1a_atomicAdd_global, 1, 256, (void*)h_out, (const void*)h_in, N);

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
