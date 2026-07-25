#include <cstdint>
#include <iostream>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void VL_int4_bitwise();
int main() {
    constexpr int N = 256;
    int *h_a=new int[N*4], *h_b=new int[N*4], *h_out=new int[N*4];
    memset(h_out, 0, N*4*sizeof(int));
    for (int i = 0; i < N*4; i++) { h_a[i]=i*0x12345; h_b[i]=i*0x67890; }
    launchKernel(VL_int4_bitwise, (N+255)/256, 256, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (h_out[i*4]!=(h_a[i*4]^h_b[i*4]) || h_out[i*4+1]!=(h_a[i*4+1]&h_b[i*4+1]) ||
            h_out[i*4+2]!=(h_a[i*4+2]|h_b[i*4+2]) || h_out[i*4+3]!=(~h_a[i*4+3]))
            { std::cout << "FAIL " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
