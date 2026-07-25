#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SHa_add_const();

int main() {
    constexpr int N = 1024, BS = 256;
    float K = 42.0f;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(SHa_add_const, (N+BS-1)/BS, BS, h_in, h_out, N, K);
    bool ok = true;
    for (int i = 0; i < N; i++)
        if (fabsf(h_out[i]-(h_in[i]+K))>1e-5f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
