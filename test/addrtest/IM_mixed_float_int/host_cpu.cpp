#include <cstdint>
#include <iostream>
#include <cmath>
#include <cstring>
#include "kernel_wrapper.h"
extern "C" void IM_mixed_float_int();
int main() {
    constexpr int N = 256;
    float *h_fin = new float[N], *h_fout = new float[N];
    int *h_iin = new int[N], *h_iout = new int[N];
    memset(h_fout, 0, N*4); memset(h_iout, 0, N*4);
    for (int i = 0; i < N; i++) { h_fin[i] = (float)(i+1)*0.01f; h_iin[i] = i*7+3; }
    launchKernel(IM_mixed_float_int, (N+255)/256, 256, h_fin, h_iin, h_fout, h_iout, N);
    bool ok = true;
    for (int i = 0; i < N; i++) {
        float fexp = h_fin[i] * 3.14159265f;
        int iexp = h_iin[i] / 7;
        if (fabsf(h_fout[i]-fexp) > fabsf(fexp)*1e-5f+1e-7f) {
            std::cout << "FAIL float at " << i << ": " << h_fout[i] << " vs " << fexp << std::endl; ok=false; break;
        }
        if (h_iout[i] != iexp) {
            std::cout << "FAIL int at " << i << ": " << h_iout[i] << " vs " << iexp << std::endl; ok=false; break;
        }
    }
    if (ok) std::cout << "TEST PASSED" << std::endl;
    delete[] h_fin; delete[] h_fout; delete[] h_iin; delete[] h_iout;
    return ok ? 0 : 1;
}
