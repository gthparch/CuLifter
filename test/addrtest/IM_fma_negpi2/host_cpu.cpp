#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void IM_fma_negpi2();
int main() {
    constexpr int N = 256;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i + 1) * 0.01f;
    launchKernel(IM_fma_negpi2, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float x = h_in[i], c = -1.570796251296997f;
        float exp = x * c + 1.0f;
        if (std::isnan(h_out[i]) || std::isinf(h_out[i]) ||
            fabsf(h_out[i] - exp) > fabsf(exp) * 1e-5f + 1e-7f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
