#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MU_sin_cos();
int main() {
    constexpr int N = 1024;
    float *h_in = new float[N], *h_s = new float[N], *h_c = new float[N];
    memset(h_s, 0, N*sizeof(float)); memset(h_c, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)i * 0.01f;
    launchKernel(MU_sin_cos, (N+255)/256, 256, h_in, h_s, h_c, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (fabsf(h_s[i] - sinf(h_in[i])) > 1e-5f || fabsf(h_c[i] - cosf(h_in[i])) > 1e-5f) {
            std::cout << "FAIL at " << i << std::endl; ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_s; delete[] h_c;
    return ok ? 0 : 1;
}
