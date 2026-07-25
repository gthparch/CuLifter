#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_shr_arith();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N], *h_out = new int[N];
    memset(h_out, 0, N*4);
    for (int i = 0; i < N; i++) h_in[i] = (i % 2 == 0) ? (i * 1000) : -(i * 1000);
    launchKernel(V_shr_arith, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int exp = h_in[i] >> 31;
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
