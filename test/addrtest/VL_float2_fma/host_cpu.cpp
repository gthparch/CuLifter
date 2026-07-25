#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_float2_fma();
int main() {
    constexpr int N = 512;
    float *h_a=new float[N*2], *h_b=new float[N*2], *h_c=new float[N*2], *h_out=new float[N*2];
    memset(h_out, 0, N*2*sizeof(float));
    for (int i = 0; i < N*2; i++) { h_a[i]=(float)(i%7+1)*0.1f; h_b[i]=(float)(i%5+1)*0.2f; h_c[i]=(float)(i%3+1)*0.3f; }
    launchKernel(VL_float2_fma, (N+255)/256, 256, h_a, h_b, h_c, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float ex = h_a[i*2]*h_b[i*2]+h_c[i*2];
        float ey = h_a[i*2+1]*h_b[i*2+1]+h_c[i*2+1];
        if (fabsf(h_out[i*2]-ex)>1e-4f || fabsf(h_out[i*2+1]-ey)>1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_out;
    return ok ? 0 : 1;
}
