#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B26c_lop3_majority();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_a = new unsigned int[N], *h_b = new unsigned int[N];
    unsigned int *h_c = new unsigned int[N], *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) {
        h_a[i] = (unsigned int)(i*0xAAAAAAAAu+1);
        h_b[i] = (unsigned int)(i*0x55555555u+3);
        h_c[i] = (unsigned int)(i*0x33333333u+5);
    }
    launchKernel(B26c_lop3_majority, (N+BS-1)/BS, BS, (unsigned int*)h_a, (unsigned int*)h_b, (unsigned int*)h_c, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = (h_a[i]&h_b[i]) | (h_b[i]&h_c[i]) | (h_a[i]&h_c[i]);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_out;
    return ok ? 0 : 1;
}
