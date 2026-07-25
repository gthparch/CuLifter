#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <iostream>
#include "kernel_wrapper.h"

extern "C" void kern_shfl_broadcast();

int main() {
    constexpr int N = 256;

    float h_in[N];
    float h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1);

    launchKernel(kern_shfl_broadcast, 1, N, h_in, h_out, N);

    // Load GPU reference
    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { std::cerr << "Cannot open ref_output.bin" << std::endl; return 1; }
    float ref[N];
    fread(ref, sizeof(float), N, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_out[i] - ref[i]) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i]
                      << " ref " << ref[i] << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
