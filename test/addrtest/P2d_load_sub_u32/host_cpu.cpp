#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2d_load_sub_u32();

int main() {
    constexpr int N = 1024, K = 3;
    uint32_t *h_data = new uint32_t[N];
    int *h_off = new int[N];
    uint32_t *h_out = new uint32_t[N];
    memset(h_out, 0, N*sizeof(uint32_t));
    for (int i = 0; i < N; i++) { h_data[i] = (uint32_t)((i%7+1)*100); h_off[i] = (i * 7 + K) % N; }

    launchKernel(P2d_load_sub_u32, (N+255)/256, 256, (uint32_t*)h_data, h_off, h_out, N, K);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int idx = h_off[i] - K;
        if (idx >= 0 && idx < N) {
            if (h_out[i] != h_data[idx]) {
                std::cout << "FAIL at " << i << std::endl;
                ok = false; break;
            }
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_data; delete[] h_off; delete[] h_out;
    return ok ? 0 : 1;
}
