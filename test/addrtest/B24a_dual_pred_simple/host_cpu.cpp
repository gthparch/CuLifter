#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B24a_dual_pred_simple();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i*3+7);
    launchKernel(B24a_dual_pred_simple, (N+BS-1)/BS, BS, (float*)h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float x = (float)i - (float)(N/2);
        float exp = (x > 0.0f && x < (float)N) ? h_in[i]+x : h_in[i];
        if (fabsf(h_out[i]-exp) > 1e-4f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
