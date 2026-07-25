#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH4_shared_reduce();

int main() {
    constexpr int N = 1024, BS = 256;
    constexpr int NBLK = (N + BS - 1) / BS;
    float *h_in = new float[N], *h_out = new float[NBLK];
    memset(h_out, 0, NBLK*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;  // each element = 1, so block sum = BS
    launchKernel(SH4_shared_reduce, NBLK, BS, h_in, h_out, N);
    bool ok = true;
    for (int b = 0; b < NBLK; b++) {
        int count = (b < NBLK-1) ? BS : (N - b*BS);
        float exp = (float)count;
        if (fabsf(h_out[b] - exp) > 0.5f) {
            std::cout << "FAIL at block " << b << ": got " << h_out[b] << " exp " << exp << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
