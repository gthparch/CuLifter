#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_vote_all();

int main() {
    constexpr int N = 256;

    int h_in[N];
    int h_out[N];
    memset(h_out, 0, sizeof(h_out));

    // Warp 0-3: all threads have in[i]=1 -> __all_sync = true
    // Warp 4-7: only lane 0 has in[i]=1, rest 0 -> __all_sync = false
    for (int i = 0; i < N; i++) {
        int warp = i / 32;
        if (warp < 4) {
            h_in[i] = 1;
        } else {
            h_in[i] = ((i % 32) == 0) ? 1 : 0;
        }
    }

    launchKernel(kern_vote_all, 1, N, h_in, h_out, N);

    // Load GPU reference
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { std::cerr << "Cannot open ref_output.bin" << std::endl; return 1; }
    int ref[N];
    fread(ref, sizeof(int), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != ref[i]) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " ref " << ref[i] << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
