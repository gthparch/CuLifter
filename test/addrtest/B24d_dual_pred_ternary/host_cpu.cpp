#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void B24d_dual_pred_ternary();

static float max3(float a, float b, float c) {
    return (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c);
}

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_a = new float[N], *h_b = new float[N];
    float *h_c = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)((i*7+3) % 100);
        h_b[i] = (float)((i*13+5) % 100);
        h_c[i] = (float)((i*19+11) % 100);
    }
    launchKernel(B24d_dual_pred_ternary, (N+BS-1)/BS, BS, (float*)h_a, (float*)h_b, (float*)h_c, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = max3(h_a[i], h_b[i], h_c[i]);
        if (fabsf(h_out[i]-exp) > 1e-5f) { std::cout << "FAIL at " << i << std::endl; ok=false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_c; delete[] h_out;
    return ok ? 0 : 1;
}
