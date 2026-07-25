#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_shfl_reduce_sum();

int main() {
    constexpr int N = 256;
    constexpr int NWARPS = N / 32;

    float h_in[N];
    float h_out[NWARPS];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    launchKernel(kern_shfl_reduce_sum, 1, N, h_in, h_out, N);

    // Load GPU reference
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { std::cerr << "Cannot open ref_output.bin" << std::endl; return 1; }
    float ref[NWARPS];
    fread(ref, sizeof(float), NWARPS, f);
    fclose(f);

    // The x86 runtime runs all 256 threads with full warp shuffle
    // synchronization, so the output should match the GPU reference.
    bool ok = true;
    for (int i = 0; i < NWARPS; i++) {
        if (fabsf(h_out[i] - ref[i]) > 1.0f) {
            std::cout << "FAIL at warp " << i << ": got " << h_out[i]
                      << " ref " << ref[i] << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
