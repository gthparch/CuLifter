#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA2a_elect_one();

int main(int argc, char** argv) {
    int h_out[1] = {-1};
    // Kernel signature is (int* out, int N): exactly two params. A spurious
    // third arg shifts N to the wrong const-bank offset, so the SASS reads
    // N=0, every lane takes `@P0 EXIT`, and out[0] is never written (-1).
    launchKernel(ISA2a_elect_one, 1, 32, (void*)h_out, 32);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 1, f);
        fclose(f);
    }

    if (h_out[0] == 0) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: got %d, expected 0\n", h_out[0]);
        return 1;
    }
    return 0;
}
