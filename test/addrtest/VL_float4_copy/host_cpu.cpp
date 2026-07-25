#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_float4_copy();
int main() {
    constexpr int N = 256;
    float *h_in = new float[N*4], *h_out = new float[N*4];
    memset(h_out, 0, N*4*sizeof(float));
    for (int i = 0; i < N*4; i++) h_in[i] = (float)(i+1);
    launchKernel(VL_float4_copy, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N*4; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
