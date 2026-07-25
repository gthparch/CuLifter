#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void S_sten5_u8_1d();

int main() {
    constexpr int N = 1024;
    uint8_t *h_in = new uint8_t[N];
    uint8_t *h_out = new uint8_t[N];
    memset(h_out, 0, N * sizeof(uint8_t));
    for (int i = 0; i < N; i++) h_in[i] = (uint8_t)((i%7)+1);

    launchKernel(S_sten5_u8_1d, (N + 255) / 256, 256, (uint8_t*)h_in, h_out, N);

    bool ok = true;
    for (int i = 2; i < N - 2; i++) {
        if (abs((int)h_out[i]-(int)(uint8_t)(h_in[i-2]+h_in[i-1]+h_in[i]+h_in[i+1]+h_in[i+2])) > 0) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
