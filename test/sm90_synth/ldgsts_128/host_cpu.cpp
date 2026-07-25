#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdio>
#include "kernel_wrapper.h"

extern "C" void ldgsts_128();

int main() {
    constexpr int N = 256;
    // float4 = 4 floats = 16 bytes
    float h_in[N * 4], h_out[N * 4];
    memset(h_out, 0, N * 4 * sizeof(float));
    for (int i = 0; i < N * 4; i++) h_in[i] = (float)(i + 1);

    launchKernel(ldgsts_128, 1, 256, h_in, h_out, N);

    FILE* f = fopen("ref_output.bin", "rb");
    if (!f) { std::cerr << "Cannot open ref_output.bin" << std::endl; return 1; }
    float ref[N * 4];
    fread(ref, sizeof(float), N * 4, f);
    fclose(f);

    bool ok = true;
    for (int i = 0; i < N * 4; i++) {
        if (fabsf(h_out[i] - ref[i]) > 1e-4f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " ref " << ref[i] << std::endl;
            ok = false;
            break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
