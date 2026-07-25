#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_float4_scale();
int main() {
    constexpr int N = 256;
    float alpha = 2.5f;
    float *h_in = new float[N*4], *h_out = new float[N*4];
    memset(h_out, 0, N*4*sizeof(float));
    for (int i = 0; i < N*4; i++) h_in[i] = (float)(i+1) * 0.01f;
    launchKernel(VL_float4_scale, (N+255)/256, 256, h_in, h_out, alpha, N);
    bool ok = true;
    for (int i = 0; i < N*4; i++) {
        float exp = h_in[i] * alpha;
        if (fabsf(h_out[i]-exp) > fabsf(exp)*1e-5f+1e-7f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
