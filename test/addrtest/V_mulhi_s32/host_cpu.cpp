#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_mulhi_s32();
int main() {
    constexpr int N = 1024;
    int *h_a = new int[N], *h_b = new int[N], *h_out = new int[N];
    memset(h_out, 0, N * 4);
    for (int i = 0; i < N; i++) { h_a[i] = (i * 12345) - 500000; h_b[i] = (i * 54321) - 300000; }
    launchKernel(V_mulhi_s32, (N+255)/256, 256, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int exp = (int)(((int64_t)h_a[i] * (int64_t)h_b[i]) >> 32);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
