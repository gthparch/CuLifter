#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH18_smem_halfword();

int main() {
    constexpr int N = 1024, BS = 256;
    short *h_in = new short[N], *h_out = new short[N];
    memset(h_out, 0, N*sizeof(short));
    for (int i = 0; i < N; i++) h_in[i] = (short)((i%41)-20);
    launchKernel(SH18_smem_halfword, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
