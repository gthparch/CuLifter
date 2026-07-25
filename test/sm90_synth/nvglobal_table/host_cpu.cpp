#include <cstdio>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void kern_nvglobal_table();

int main() {
    constexpr int N = 256;
    unsigned int h_out[N];
    int h_idx[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_idx[i] = i * 3 + 1;

    launchKernel(kern_nvglobal_table, 1, 256, h_out, h_idx, N);

    static const unsigned int magic[8] = {
        0x1111aaaau, 0x2222bbbbu, 0x3333ccccu, 0x4444ddddu,
        0x5555eeeeu, 0x6666ffffu, 0x77770123u, 0x88884567u,
    };
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int want = magic[h_idx[i] & 7];  // zero_table adds 0
        if (h_out[i] != want) {
            printf("FAIL at %d: got 0x%08x want 0x%08x\n", i, h_out[i], want);
            ok = false;
            break;
        }
    }
    printf(ok ? "TEST PASSED\n" : "TEST FAILED\n");
    return ok ? 0 : 1;
}
