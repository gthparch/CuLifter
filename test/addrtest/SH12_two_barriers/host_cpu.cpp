#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void SH12_two_barriers();

int main() {
    constexpr int N = 1024, BS = 256;
    float *h_a = new float[N], *h_b = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) { h_a[i] = (float)(i%7+1); h_b[i] = (float)(i%5+1); }
    launchKernel(SH12_two_barriers, (N+BS-1)/BS, BS, h_a, h_b, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = h_a[i] + h_b[i];
        if (fabsf(h_out[i] - exp) > 1e-5f) {
            std::cout << "FAIL at " << i << ": got " << h_out[i] << " exp " << exp << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_a; delete[] h_b; delete[] h_out;
    return ok ? 0 : 1;
}
