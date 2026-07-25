#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void ST_float4_ptr();
int main() {
    constexpr int N = 64;
    float h_in[N*4], h_out[N*4];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N*4; i++) h_in[i] = (float)(i+1);
    launchKernel(ST_float4_ptr, (N+63)/64, 64, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N*4; i++)
        if (fabsf(h_out[i] - h_in[i]*2.0f) > 1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
