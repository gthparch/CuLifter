#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_float4_dot();
int main() {
    constexpr int N = 256;
    float *h_a=new float[N*4], *h_b=new float[N*4], *h_out=new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N*4; i++) { h_a[i]=(float)(i%7+1)*0.1f; h_b[i]=(float)(i%5+1)*0.1f; }
    launchKernel(VL_float4_dot, (N+255)/256, 256, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_a[i*4]*h_b[i*4]+h_a[i*4+1]*h_b[i*4+1]+h_a[i*4+2]*h_b[i*4+2]+h_a[i*4+3]*h_b[i*4+3];
        if (fabsf(h_out[i]-exp) > fabsf(exp)*1e-4f+1e-6f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
