#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_store_u8();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N]; uint8_t *h_out = new uint8_t[N];
    memset(h_out, 0, N);
    for (int i = 0; i < N; i++) h_in[i] = i * 7 + 3;
    launchKernel(V_store_u8, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint8_t exp = (uint8_t)(h_in[i] & 0xFF);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
