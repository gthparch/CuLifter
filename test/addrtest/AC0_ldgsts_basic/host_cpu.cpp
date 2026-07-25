#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void AC0_ldgsts_basic();
int main() {
    constexpr int N = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i+1);
    launchKernel(AC0_ldgsts_basic, 1, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - h_in[i]*2.0f) > 1e-5f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
