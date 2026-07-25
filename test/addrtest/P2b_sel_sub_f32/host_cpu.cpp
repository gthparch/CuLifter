#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2b_sel_sub_f32();

int main() {
    constexpr int N = 1024, offset = 4;
    float *h_A = new float[N], *h_B = new float[N], *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) { h_A[i] = (float)((i%7+1)*0.1f); h_B[i] = (float)((i%5+1)*0.2f); }

    int selector = 1;
    launchKernel(P2b_sel_sub_f32, (N+255)/256, 256, (float*)h_A, (float*)h_B, h_out, N, selector, offset);

    bool ok = true;
    for (int i = offset; i < N; i++) {
        float expected = h_B[i - offset];
        if (h_out[i] != expected) {
            std::cout << "FAIL at " << i << std::endl;
            ok = false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_A; delete[] h_B; delete[] h_out;
    return ok ? 0 : 1;
}
