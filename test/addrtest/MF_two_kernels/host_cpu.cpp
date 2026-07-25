#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MF_kernel_simple();
int main() {
    constexpr int N = 256;
    float h_in[N], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1);
    launchKernel(MF_kernel_simple, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - h_in[i]*2.0f) > 1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
