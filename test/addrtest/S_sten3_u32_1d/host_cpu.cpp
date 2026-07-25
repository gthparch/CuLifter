#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_sten3_u32_1d();

int main() {
    constexpr int N = 1024;
    uint32_t *h_in = new uint32_t[N];
    uint32_t *h_out = new uint32_t[N];
    memset(h_out, 0, N * sizeof(uint32_t));
    for (int i = 0; i < N; i++) h_in[i] = (uint32_t)((i%7)+1)*100;

    launchKernel(S_sten3_u32_1d, (N + 255) / 256, 256, (uint32_t*)h_in, h_out, N);

    bool ok = true;
    for (int i = 1; i < N - 1; i++) {
        if (abs((int)h_out[i]-(int)(uint32_t)(h_in[i-1] + h_in[i] + h_in[i+1])) > 0) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
