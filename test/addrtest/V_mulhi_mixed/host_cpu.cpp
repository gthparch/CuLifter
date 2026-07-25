#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_mulhi_mixed();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N], *h_out = new int[N];
    memset(h_out, 0, N * 4);
    for (int i = 0; i < N; i++) h_in[i] = i * 17 - 500;
    launchKernel(V_mulhi_mixed, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int exp = (int)(((int64_t)h_in[i] * 0x55555556LL) >> 32);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
