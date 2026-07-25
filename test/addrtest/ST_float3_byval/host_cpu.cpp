#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void ST_float3_byval();
int main() {
    constexpr int N = 256;
    float h_out[N]; memset(h_out, 0, sizeof(h_out));
    // float3 = 12 bytes (x, y, z)
    float offset[3] = {1.0f, 2.0f, 3.0f};
    // Pass struct by value: write 12 bytes to const_mem
    launchKernel(ST_float3_byval, (N+255)/256, 256, h_out, offset[0], offset[1], offset[2], N);
    bool ok = true;
    float sum = 1.0f + 2.0f + 3.0f;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i] - (sum + (float)i)) > 1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
