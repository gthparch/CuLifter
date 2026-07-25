#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHt_u8_copy();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned char *h_in = new unsigned char[N], *h_out = new unsigned char[N];
    memset(h_out, 0, N*sizeof(unsigned char));
    for (int i = 0; i < N; i++) h_in[i] = (unsigned char)(i%200);
    launchKernel(SHt_u8_copy, (N+BS-1)/BS, BS, (unsigned char*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i]!=h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
