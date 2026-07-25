#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MF_add();
int main() {
    constexpr int N = 256;
    float h_a[N], h_b[N], h_c[N];
    memset(h_c, 0, sizeof(h_c));
    for (int i = 0; i < N; i++) { h_a[i] = (float)(i+1); h_b[i] = (float)(i*2+1); }
    launchKernel(MF_add, (N+255)/256, 256, h_a, h_b, h_c, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_c[i] - (h_a[i]+h_b[i])) > 1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
