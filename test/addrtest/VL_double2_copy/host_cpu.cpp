#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_double2_copy();
int main() {
    constexpr int N = 512;
    double *h_in = new double[N*2], *h_out = new double[N*2];
    memset(h_out, 0, N*2*sizeof(double));
    for (int i = 0; i < N*2; i++) h_in[i] = (double)(i+1) * 0.001;
    launchKernel(VL_double2_copy, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N*2; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
