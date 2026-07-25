#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void ST_int4_ptr();
int main() {
    constexpr int N = 64;
    int h_in[N*4], h_out[N];
    memset(h_out, 0, sizeof(h_out));
    for (int i = 0; i < N*4; i++) h_in[i] = i + 1;
    launchKernel(ST_int4_ptr, (N+63)/64, 64, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int exp = h_in[i*4] + h_in[i*4+1] + h_in[i*4+2] + h_in[i*4+3];
        if (h_out[i] != exp) { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    return ok ? 0 : 1;
}
