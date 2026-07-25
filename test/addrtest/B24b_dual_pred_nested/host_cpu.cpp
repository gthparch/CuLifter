#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B24b_dual_pred_nested();

int main() {
    constexpr int N = 1024, BS = 256;
    int *h_flags = new int[N];
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) { h_flags[i] = (i % 3 == 0) ? 1 : 0; h_in[i] = (float)(i+1); }
    launchKernel(B24b_dual_pred_nested, (N+BS-1)/BS, BS, (int*)h_flags, (float*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_flags[i] ? h_in[i]*2.0f : h_in[i];
        if (fabsf(h_out[i]-exp) > 1e-4f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_flags; delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
