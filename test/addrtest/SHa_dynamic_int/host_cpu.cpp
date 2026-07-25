#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHa_dynamic_int();

int main() {
    constexpr int N = 1024, BS = 256;
    int *h_in = new int[N], *h_out = new int[N];
    memset(h_out, 0, N*sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = i*3+7;
    launchKernel(SHa_dynamic_int, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
