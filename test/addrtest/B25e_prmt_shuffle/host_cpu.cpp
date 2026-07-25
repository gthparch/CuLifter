#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B25e_prmt_shuffle();

static unsigned int cpu_byte_perm(unsigned int a, unsigned int b, unsigned int sel) {
    unsigned char src[8];
    src[0] = (unsigned char)(a); src[1] = (unsigned char)(a>>8);
    src[2] = (unsigned char)(a>>16); src[3] = (unsigned char)(a>>24);
    src[4] = (unsigned char)(b); src[5] = (unsigned char)(b>>8);
    src[6] = (unsigned char)(b>>16); src[7] = (unsigned char)(b>>24);
    unsigned int r = 0;
    for (int j = 0; j < 4; j++) {
        unsigned int idx = (sel >> (j*4)) & 0x7u;
        r |= ((unsigned int)src[idx]) << (j*8);
    }
    return r;
}

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_a = new unsigned int[N], *h_b = new unsigned int[N];
    unsigned int *h_out = new unsigned int[N];
    memset(h_out, 0, N*sizeof(unsigned int));
    for (int i = 0; i < N; i++) { h_a[i] = (unsigned int)(i*0x01030507u+1); h_b[i] = (unsigned int)(i*0x02040608u+3); }
    launchKernel(B25e_prmt_shuffle, (N+BS-1)/BS, BS, (unsigned int*)h_a, (unsigned int*)h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        unsigned int exp = cpu_byte_perm(h_a[i], h_b[i], 0x05040100u);
        if (h_out[i] != exp) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
