#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B26d_lop3_mux();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_sel = new unsigned int[N], *h_a = new unsigned int[N];
    unsigned int *h_b = new unsigned int[N], *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) {
        h_sel[i] = (unsigned int)(i*0xF0F0F0F0u+0x0F0F0F0Fu);
        h_a[i] = (unsigned int)(i*0x12345678u+1);
        h_b[i] = (unsigned int)(i*0x87654321u+2);
    }
    launchKernel(B26d_lop3_mux, (N+BS-1)/BS, BS, (unsigned int*)h_sel, (unsigned int*)h_a, (unsigned int*)h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = (h_sel[i] & h_a[i]) | (~h_sel[i] & h_b[i]);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_sel; delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
