#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MU_exp_chain();
int main() {
    constexpr int N = 1024;
    float *h_a = new float[N], *h_b = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = (float)((i%7)-3)*0.3f; h_b[i] = (float)((i%5)-2)*0.4f; }
    launchKernel(MU_exp_chain, (N+255)/256, 256, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp_val = expf(h_a[i]) * expf(h_b[i]);
        if (fabsf(h_out[i]-exp_val) > fabsf(exp_val)*1e-3f+1e-6f) {
            std::cout << "FAIL at " << i << std::endl; ok=false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
