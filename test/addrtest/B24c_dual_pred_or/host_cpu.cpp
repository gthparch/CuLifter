#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B24c_dual_pred_or();

int main() {
    constexpr int N = 1024, BS = 256;
    int total = N + 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*2+3);
    launchKernel(B24c_dual_pred_or, (total+BS-1)/BS, BS, (float*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_in[i] + 1.0f;
        if (fabsf(h_out[i]-exp) > 1e-5f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
