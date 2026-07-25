#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHb_bs32_copy();

int main() {
    constexpr int N = 1024, BS = 32;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SHb_bs32_copy, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
