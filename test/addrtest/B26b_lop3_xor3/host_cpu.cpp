#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B26b_lop3_xor3();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_a = new unsigned int[N], *h_b = new unsigned int[N];
    unsigned int *h_c = new unsigned int[N], *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) {
        h_a[i] = (unsigned int)(i*0xDEAD0001u+7);
        h_b[i] = (unsigned int)(i*0xBEEF0002u+11);
        h_c[i] = (unsigned int)(i*0xCAFE0003u+13);
    }
    launchKernel(B26b_lop3_xor3, (N+BS-1)/BS, BS, (unsigned int*)h_a, (unsigned int*)h_b, (unsigned int*)h_c, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = h_a[i] ^ h_b[i] ^ h_c[i];
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_out;
    return ok ? 0 : 1;
}
