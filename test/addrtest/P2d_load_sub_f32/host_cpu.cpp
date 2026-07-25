#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"

extern "C" void P2d_load_sub_f32();

int main() {
    constexpr int N = 1024, K = 3;
    float *h_data = new float[N];
    int *h_off = new int[N];
    float *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) { h_data[i] = (float)((i%7+1)*0.1f); h_off[i] = (i * 7 + K) % N; }

    launchKernel(P2d_load_sub_f32, (N+255)/256, 256, (float*)h_data, h_off, h_out, N, K);

    bool ok = true;
    for (int i = 0; i < N; i++) {
        int idx = h_off[i] - K;
        if (idx >= 0 && idx < N) {
            if (h_out[i] != h_data[idx]) {
                std::cout << "FAIL at " << i << std::endl;
                ok = false; break;
            }
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_data; delete[] h_off; delete[] h_out;
    return ok ? 0 : 1;
}
