#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_store_u16();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N]; uint16_t *h_out = new uint16_t[N];
    memset(h_out, 0, N*2);
    for (int i = 0; i < N; i++) h_in[i] = i * 7 + 3;
    launchKernel(V_store_u16, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint16_t exp = (uint16_t)(h_in[i] & 0xFFFF);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
