#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_mulhi_u32();
int main() {
    constexpr int N = 1024;
    uint32_t *h_a = new uint32_t[N], *h_b = new uint32_t[N], *h_out = new uint32_t[N];
    memset(h_out, 0, N * 4);
    for (int i = 0; i < N; i++) { h_a[i] = (uint32_t)(i * 12345 + 67890); h_b[i] = (uint32_t)(i * 54321 + 11111); }
    launchKernel(V_mulhi_u32, (N+255)/256, 256, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint32_t exp = (uint32_t)(((uint64_t)h_a[i] * (uint64_t)h_b[i]) >> 32);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
