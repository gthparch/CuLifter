#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH17_smem_type_pun();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_in = new float[N];
    uint32_t *h_out = new uint32_t[N];
    memset(h_out, 0, N*sizeof(uint32_t));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SH17_smem_type_pun, (N+BS-1)/BS, BS, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        uint32_t expected; memcpy(&expected, &h_in[i], 4);
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << ": got 0x" << std::hex << h_out[i]
                      << " exp 0x" << expected << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
