#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void MU_rsq();
int main() {
    constexpr int N = 1024;
    float *h_in = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = (float)(i % 100 + 1) * 0.1f;
    launchKernel(MU_rsq, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float x = h_in[i];
        float exp_val = 1.0f / sqrtf(x);
        if (std::isnan(h_out[i]) || std::isinf(h_out[i]) || fabsf(h_out[i] - exp_val) > fabsf(exp_val) * 1e-3f + 1e-6f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp_val << " (in=" << x << ")" << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
