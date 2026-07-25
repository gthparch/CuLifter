#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B28d_int_divmod_combo();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_in = new unsigned int[N], *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)(i*23+9);
    launchKernel(B28d_int_divmod_combo, (N+BS-1)/BS, BS, (unsigned int*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i] != h_in[i]) { std::cout << "FAIL at " << i << " got=" << h_out[i] << " exp=" << h_in[i] << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
