#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH8_smem_no_cga();

int main() {
    constexpr int N = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SH8_smem_no_cga, 1, N, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
