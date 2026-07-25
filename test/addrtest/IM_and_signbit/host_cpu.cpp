#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void IM_and_signbit();
int main() {
    constexpr int N = 256;
    int *h_in = new int[N], *h_out = new int[N];
    memset(h_out, 0, N*sizeof(int));
    for (int i = 0; i < N; i++) h_in[i] = i * 7 + 3;
    launchKernel(IM_and_signbit, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int x = h_in[i];
        int exp = (int)((unsigned)h_in[i] & 0x80000000u);
        if (h_out[i] != exp) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
