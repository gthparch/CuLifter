#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B25c_prmt_pack_bytes();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned char *h_in = new unsigned char[N*4];
    unsigned int *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N*4; i++) h_in[i] = (unsigned char)((i*7+3)%256);
    launchKernel(B25c_prmt_pack_bytes, (N+BS-1)/BS, BS, (unsigned char*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        int base = i*4;
        unsigned int exp = (unsigned int)h_in[base] | ((unsigned int)h_in[base+1]<<8)
                         | ((unsigned int)h_in[base+2]<<16) | ((unsigned int)h_in[base+3]<<24);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
