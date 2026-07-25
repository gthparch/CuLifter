#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B25b_prmt_extract_half();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_in = new unsigned int[N];
    int *h_out = new int[N];
    memset(h_out, 0, N*sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)((i*37+11) | ((i*53+7) << 16));
    launchKernel(B25b_prmt_extract_half, (N+BS-1)/BS, BS, (unsigned int*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        int exp = (int)(short)(x & 0xFFFFu) + (int)(short)(x >> 16);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << " got=" << h_out[i] << " exp=" << exp << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
