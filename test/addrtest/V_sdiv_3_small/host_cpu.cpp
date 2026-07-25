#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_sdiv_3_small();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N], *h_q = new int[N], *h_r = new int[N];
    memset(h_q, 0, N*4); memset(h_r, 0, N*4);
    for (int i = 0; i < N; i++) h_in[i] = i % 11;
    launchKernel(V_sdiv_3_small, (N+255)/256, 256, h_in, h_q, h_r, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_q[i] != h_in[i]/3 || h_r[i] != h_in[i]%3) {
            std::cout << "FAIL at " << i << ": in=" << h_in[i] << " q=" << h_q[i] << "(exp " << h_in[i]/3 << ") r=" << h_r[i] << "(exp " << h_in[i]%3 << ")" << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_q; delete[] h_r;
    return ok ? 0 : 1;
}
