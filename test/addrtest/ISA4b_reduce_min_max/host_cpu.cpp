#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA4b_reduce_min_max();

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32];
    int h_out[2] = {0, 0};
    for (int i = 0; i < N; i++) h_in[i] = (i * 7 + 13) % 100 - 50;

    // The x86 wrapper runs all 32 warp lanes concurrently, so
    // __reduce_min_sync / __reduce_max_sync are REAL warp-wide reductions
    // (matching the GPU): every lane receives the min/max over all members and
    // tid==0 stores them. Expected = min/max over the warp (-48 / 47), not the
    // old single-lane identity (in[0]=-37).
    int exp_min = h_in[0], exp_max = h_in[0];
    for (int i = 1; i < N; i++) {
        if (h_in[i] < exp_min) exp_min = h_in[i];
        if (h_in[i] > exp_max) exp_max = h_in[i];
    }
    launchKernel(ISA4b_reduce_min_max, 1, 32, (void*)h_out, (void*)(h_out+1), (const void*)h_in, N);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), 2, f);
        fclose(f);
    }

    if (h_out[0] == exp_min && h_out[1] == exp_max) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED: min=%d, max=%d, expected min=%d max=%d\n",
               h_out[0], h_out[1], exp_min, exp_max);
        return 1;
    }
    return 0;
}
