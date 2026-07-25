#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_vote_any();

int main() {
    constexpr int N = 256;

    int h_in[N];
    int h_out[N];
    memset(h_out, 0, sizeof(h_out));

    // Only lane 0 of each warp has a positive value
    for (int i = 0; i < N; i++) {
        h_in[i] = ((i % 32) == 0) ? 1 : 0;
    }

    launchKernel(kern_vote_any, 1, N, h_in, h_out, N);

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
