#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B26e_prmt_lop3_combo();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_in = new unsigned int[N], *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)(i*0x01030507u+0x0B0D1113u);
    launchKernel(B26e_prmt_lop3_combo, (N+BS-1)/BS, BS, (unsigned int*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int x = h_in[i];
        unsigned int b0 = x&0xFFu, b1 = (x>>8)&0xFFu, b2 = (x>>16)&0xFFu;
        unsigned int exp = (b0&b1)|(b1&b2)|(b0&b2);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
