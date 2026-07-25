#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_saxpy_vec4();
int main() {
    constexpr int N = 256;
    float alpha = 2.5f;
    float *h_x=new float[N*4], *h_y=new float[N*4], *h_y_orig=new float[N*4];
    for (int i = 0; i < N*4; i++) { h_x[i]=(float)(i%17+1)*0.1f; h_y[i]=(float)(i%13+1)*0.2f; }
    memcpy(h_y_orig, h_y, N*4*sizeof(float));
    launchKernel(VL_saxpy_vec4, (N+255)/256, 256, h_x, h_y, alpha, N);
    bool ok = true;
    for (int i = 0; i < N*4; i++) {
        float exp = h_y_orig[i] + alpha * h_x[i];
        if (fabsf(h_y[i]-exp) > 1e-4f) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_x; delete[] h_y; delete[] h_y_orig;
    return ok ? 0 : 1;
}
