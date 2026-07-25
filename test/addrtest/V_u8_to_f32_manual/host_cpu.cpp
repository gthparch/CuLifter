#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void V_u8_to_f32_manual();
int main() {
    constexpr int N = 1024;
    int *h_in = new int[N]; float *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    for (int i = 0; i < N; i++) h_in[i] = i * 7 + 3;
    launchKernel(V_u8_to_f32_manual, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float exp = (float)(h_in[i] & 0xFF);
        if (fabsf(h_out[i] - exp) > 0.5f) { std::cout << "FAIL at " << i << std::endl; ok = false; break; }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
