#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2c_neg_step_u32();

int main() {
    constexpr int N = 1024, step = 5;
    uint32_t *h_in = new uint32_t[N], *h_out = new uint32_t[N];
    memset(h_out, 0, N*sizeof(uint32_t));
    for (int i = 0; i < N; i++) h_in[i] = (uint32_t)((i%7+1)*100);

    launchKernel(P2c_neg_step_u32, (N+255)/256, 256, (uint32_t*)h_in, h_out, N, step);

    bool ok = true;
    for (int i = step; i < N; i++) {
        if (h_out[i] != h_in[i - step]) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
