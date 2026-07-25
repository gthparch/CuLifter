#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void IM_int_as_float();
int main() {
    constexpr int N = 256;
    int *h_in = new int[N]; float *h_out = new float[N];
    memset(h_out, 0, N*sizeof(float));
    // Use valid float bit patterns as int inputs
    for (int i = 0; i < N; i++) {
        float v = (float)(i+1) * 0.1f;
        memcpy(&h_in[i], &v, 4);
    }
    launchKernel(IM_int_as_float, (N+255)/256, 256, h_in, h_out, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float v; memcpy(&v, &h_in[i], 4);
        float exp = v * 2.0f;
        if (fabsf(h_out[i]-exp) > fabsf(exp)*1e-5f+1e-7f) {
            std::cout << "FAIL at " << i << std::endl; ok=false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_in; delete[] h_out;
    return ok ? 0 : 1;
}
