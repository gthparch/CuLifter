#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B25d_i2f_s16_cast();

int main() {
    constexpr int N = 1024, BS = 256;
    unsigned int *h_in = new unsigned int[N];
    float *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (unsigned int)((i*97+13) & 0xFFFFu);
    launchKernel(B25d_i2f_s16_cast, (N+BS-1)/BS, BS, (unsigned int*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = (float)(short)(h_in[i] & 0xFFFFu);
        if (fabsf(h_out[i]-exp) > 1e-5f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
