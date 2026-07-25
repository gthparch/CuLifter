#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B28e_viadd_sat();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned char *h_a = new unsigned char[N], *h_b = new unsigned char[N];
    unsigned char *h_out = new unsigned char[N];
    memset(h_out, 0, N*sizeof(unsigned char));
    for (int i = 0; i < N; i++) { h_a[i] = (unsigned char)(i%200); h_b[i] = (unsigned char)((i*3+7)%200); }
    launchKernel(B28e_viadd_sat, (N+BS-1)/BS, BS, (unsigned char*)h_a, (unsigned char*)h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int sum = (unsigned int)h_a[i] + (unsigned int)h_b[i];
        unsigned char exp = (unsigned char)(sum < 255u ? sum : 255u);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
