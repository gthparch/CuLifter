#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "kernel_wrapper.h"
extern "C" void ISA3a_match_any();

int main(int argc, char** argv) {
    const int N = 32;
    int h_in[32], h_out[32];
    for (int i = 0; i < 8;  i++) h_in[i] = 10;
    for (int i = 8; i < 16; i++) h_in[i] = 20;
    for (int i = 16; i < 32; i++) h_in[i] = 30;

    // The x86 wrapper runs all 32 warp lanes concurrently, so
    // __match_any_sync is a REAL cross-lane match (matching the GPU): each lane
    // gets the mask of lanes sharing its value, and stores popc(mask). Expected
    // per lane = how many lanes hold the same value (8 / 8 / 16 here), not the
    // old single-lane identity (popc=1).
    int expected[32];
    for (int i = 0; i < N; i++) {
        int cnt = 0;
        for (int j = 0; j < N; j++) if (h_in[j] == h_in[i]) cnt++;
        expected[i] = cnt;
    }

    launchKernel(ISA3a_match_any, 1, 32, (void*)h_out, (const void*)h_in, N);

    if (argc > 1 && strcmp(argv[1], "--dump") == 0) {
        FILE* f = fopen("ref_output.bin", "wb");
        fwrite(h_out, sizeof(int), N, f);
        fclose(f);
    }

    int pass = 1;
    for (int i = 0; i < N; i++) if (h_out[i] != expected[i]) { pass = 0; break; }

    if (pass) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED\n");
        for (int i = 0; i < N; i++) printf("  [%d] = %d (exp %d)\n", i, h_out[i], expected[i]);
        return 1;
    }
    return 0;
}
